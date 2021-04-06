"""
  Prototype of RNN and its variants,
    parameter typing and description is omitted as they are self-explanatory
    places with comments worth particular attention
  
  Code comes from Book <Dive Into Deep Learning> Pytorch version
  You may refer to the followings:
    Pytorch Chinese version : https://tangshusen.me/Dive-into-DL-PyTorch/#/
    Pytorch English version : https://github.com/dsgiitr/d2l-pytorch
    
  Created by Yukun, Jiang on 06/04/2021.
  Copyright © 2021 Yukun, Jiang. All rights reserved.
"""


import random
import time
import math
import zipfile
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

class RNNModel(nn.Module):
    """
    The prototype of RNN model
    it can take in an arbitrary recurrent neural network layer: vanilla, GRU, LSTM, etc.
    """
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None
    
    def forward(self, inputs, state):
        # input: (batch, seq_len)
        # first get one-hot encoding
        # X is a list
        X = to_onehot(inputs, self.vocab_size)
        # stack X in term of time step
        Y, self.state = self.rnn(torch.stack(X), state)
        # dense layer first change Y into shape (num_steps * batch_size, num_hiddens)
        # then output in shape (num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state
        
""" use JayChou's lyric for demonstration """
def load_data_jay_lyrics():
    with zipfile.ZipFile("./jaychou_lyrics.txt.zip") as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # for convenience, replace line breaker as space
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # only take the first 10k for training
    corpus_chars = corpus_chars[0:10000]
    # build mapping from character to index and vice versa
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

""" Random sampling """
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # minus 1 because output y's index is input x's index plus 1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    
    # return a seq of length=num_steps starting from pos
    def _data(pos):
        return corpus_indices[pos : pos+num_steps]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # every time read random sample of size=batch_size
        i = i * batch_size
        batch_indices = example_indices[i : i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

""" Consecutive sampling """
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0 : batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i+num_steps]
        Y = indices[:, i+1 : i+1+num_steps]
        yield X, Y

def one_hot(x, n_class, dtype=torch.float32):
    # x shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1,1), 1)
    return res

def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:,i], n_class) for i in range(X.shape[1])]

""" Avoid gradient explosion """
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data **2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]] # output will record prefix + output from rnn
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1,1)
        if state is not None:
            if isinstance(state, tuple): # LSTM
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                 corpus_indices, idx_to_char, char_to_idx,
                                 num_epochs, num_steps, lr, clipping_theta,
                                 batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    start = time.time()
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # consecutive sampling
        
        for X, Y in data_iter:
            if state is not None:
                # to detach from last computational graph
                # because last graph is already destroyed
                if isinstance(state, tuple): # LSTM
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # output: shape (num_steps * batch_size, vocab_size)
            (output, state) = model(X, state)
            # shape of Y is originally (batch_size, num_steps),
            # transpose to vector of length batch_size * num_steps
            y = torch.transpose(Y, 0, 1).contiguous().view(-1).long()
            l = loss(output, y)
            
            optimizer.zero_grad()
            l.backward()
            # grad clipping
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        try:
            perplexity = math.exp(l_sum/n)
        except OverflowError:
            perplexity = float('inf')
        
        if (epoch + 1) % pred_period == 0:
            print('epoch: %d, perplexity: %f, time so far: %.2f sec'% (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device,idx_to_char,
                    char_to_idx))

def main(RNN_type):
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    num_hiddens, num_epochs, batch_size, lr, clipping_theta = 256, 200, 32, 1e-2, 1e-2 # notice the learning rate
    batch_size, num_steps, device, pred_period, pred_len, prefixes = 2, 35, 'cpu', 50, 50, ['分开', '不分开']
    
    # define RNN layer
    if RNN_type == "vanilla":
        rnn_layer = nn.RNN(input_size=vocab_size, hidden_size = num_hiddens)
    elif RNN_type == "gru":
        rnn_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
    elif RNN_type == "lstm":
        rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
    model = RNNModel(rnn_layer, vocab_size).to(device)
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
if __name__ == "__main__":
    main(RNN_type="vanilla")

