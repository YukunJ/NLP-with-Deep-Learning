"""
  Prototype of CNN and its variants,
    parameter typing and description is omitted as they are self-explanatory
    places with comments worth particular attention
  
  Recall the formula for CNN size:
  If denote input size n_i, kernel n_k, padding n_p, stride n_s,
  then after the convolutional layer, the size s is
    s = floor((n_h-n_k+n_p+n_s)/n_s)
  Also notice that in pytorch, the padding is one-sided,
  so in calculation may need to douuble it (ex. "padding = 2" means here n_p=4)
  
  The CNN network Prototype here includes:
  1. LeNet
  2. LeNet with Batch Normalization
  3. AlexNet
  4. VGG
  5. NIN
  6. GoogLeNet
  7. ResNet
  8. DenseNet
  
  Code comes from Book <Dive Into Deep Learning> Pytorch version
  You may refer to the followings:
    Pytorch Chinese version : https://tangshusen.me/Dive-into-DL-PyTorch/#/
    Pytorch English version : https://github.com/dsgiitr/d2l-pytorch
    
  Created by Yukun, Jiang on 20/04/2021.
  Copyright © 2021 Yukun, Jiang. All rights reserved.
"""
import sys
import time
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from utils import *

class LeNet(nn.Module):
    """ By Yann LeCun, Violet Pride! """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output

class LeNet_normalized(nn.Module):
    def __init__(self):
        super(LeNet_normalized, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, (padding)
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # reduce Conv kern size, use padding=2, and more out_channels
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 3 Conv in serial, smaller kernel, more out_channels
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # use dropout in fully-connected layer, avoid over-fitting
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # shrink the size by half
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # Convolutional Layer
    for i, (num_conv, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_conv, in_channels, out_channels))
    # Fullly-Connected Layer
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                      nn.Linear(fc_features, fc_hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(fc_hidden_units, fc_hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(fc_hidden_units, 10)
                                      ))
    return net

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk
    
def get_nin_net():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        GlobalAvgPool2d(),
        FlattenLayer()
    )
    return net
    
class Inception(nn.Module):
    # c1 - c4 is the number of out_channels
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # route 1: 1x1 conv
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # route 2: 1x1 conv + 3x3 conv
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # route 3: 1x1 conv + 5x5 conv
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # route 4: 3x3 MaxPool + 1x1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4), dim=1) # concat on out_channels

def get_GoogLeNet():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(
        Inception(192, 64, (96,128), (16,32), 32),
        Inception(256, 128, (128,192), (32,96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(
        Inception(480, 192, (96,208), (16,48), 64),
        Inception(512, 160, (112,224), (24,64), 64),
        Inception(512, 128, (128,256), (24,64), 64),
        Inception(512, 112, (144,288), (32,64), 64),
        Inception(528, 256, (160,320), (32,128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(
        Inception(832, 256, (160,320), (32,128), 128),
        Inception(832, 384, (192,384), (48,128), 128),
        GlobalAvgPool2d())
    
    net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024,10))
    return net

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
def get_ResNet():
    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    # add global avg pool and fully-connected layer
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512,10)))
    return net
    
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock,self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X,Y), dim=1) # concat on out_channels
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

def get_DenseNet():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlock_%d" % i, DB)
        # update the out_channels
        num_channels = DB.out_channels
        # add halving transition Layer in between
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))
    return net
    
def main(net):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if net == "LeNet":
        print("Training LeNet")
        net = LeNet()
        batch_size = 256
        resize = None
    elif net == "AlexNet":
        print("Training AlexNet")
        net = AlexNet()
        batch_size = 128
        resize = 224
    elif net == "VGG":
        print("Training VGG")
        fc_features = 512 * 7 * 7
        fc_hidden_units = 4096
        ratio = 8
        small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),(2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
        net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
        batch_size = 64
        resize = 224
    elif net == "NIN":
        print("Training NIN")
        net = get_nin_net()
        batch_size = 128
        resize = 224
    elif net == "GoogLeNet":
        print("Training GoogLeNet")
        net = get_GoogLeNet()
        batch_size = 128
        resize = 96
    elif net == "LeNet_normalized":
        print("Training LeNet with Batch Normalization")
        net = LeNet_normalized()
        batch_size = 256
        resize = None
    elif net == "ResNet":
        print("Training ResNet")
        net = get_ResNet()
        batch_size = 256
        resize=96
    elif net == "DenseNet":
        print("Training DenseNet")
        net = get_DenseNet()
        batch_size = 256
        resize = 96
    else:
        print("Invalid Network Name, please check again!")
        sys.exit(0)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=resize)
    train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        
if __name__ == "__main__":
    net_input = input("What kind of Network do you want?\n")
    main(net = net_input)
    
