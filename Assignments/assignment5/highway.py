#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        """
        Init Highway model : contains 2 Linear Layer
        @param word_embed_size (int): word Embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate = nn.Linear(word_embed_size, word_embed_size, bias=True)
        
    def forward(self, source):
        """
        Take a mini-batch of the output from convolutional net and go through the Highway layer
        @param source(Tensor) :  of shape (batch_size, word_embed_size)
        @returns : highway (Tensor) of shape (batch_size, word_embed_size)
        """
        proj = F.relu(self.proj(source))
        gate = torch.sigmoid(self.gate(source))
        highway = gate * proj + (1-gate) * source
        return highway

### END YOUR CODE 

