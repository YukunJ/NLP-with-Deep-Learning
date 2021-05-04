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
### YOUR CODE HERE for part 1i

class CNN(nn.Module):
    def __init__(self, input_channel_count, output_channel_count, K=5):
        """
        Init CNN network: contains Conv1d and MaxPool
        @param input_channel_count (int) : the number of input_channel
        @param ouput_channel_count (int) : the number of output_channel, i.e. filter
        @param K (int) : kernel size of the convolution, default=5
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channel_count, output_channel_count, K)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    
    def forward(self, source):
        """
        Map from x_reshaped to x_conv_out
        @param source (Tensor) : of shape (batch_size, e_char, m_word)
        @returns conv_out (Tensor) : of shape (batch_size, e_word)
        """
        conv1d = F.relu(self.conv1d(source))
        conv_out = self.maxpool(conv1d).squeeze(dim=2)
        return conv_out
        

### END YOUR CODE

