## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
## https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


import numpy as np
import scipy.misc as misc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t 


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1,10, kernel_size=3)
        self.conv2 = nn.Conv3d(10,20, kernel_size=3)
        self.conv3 = nn.Conv3d(20,40, kernel_size=2)
#        self.conv4 = nn.Conv3d(40,80, kernel_size=3)   
        self.drop = nn.Dropout3d()
        
        self.fc1 = nn.Linear(40*6*6*6, 1280)   # 4x4x4x80
        self.fc2 = nn.Linear(1280, 512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=None, padding=0)
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)

    def forward(self,x):
        x = self.batchnorm(x)
        x = self.relu(self.pool(self.conv1(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv2(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv3(x)))
        x = self.drop(x)
        
        print('after the convolutional layers, the shape is:'.format(x.shape))   # ([200, 40, 6, 6, 6])
        print(x.shape)
#        X = self.relu(self.pool(self.conv4(x)))
#        x = self.drop(x)
        x = x.view(-1, 40*6*6*6)
        print('before fc-layers, the shape is:'.format(x.shape))  ##
        print(x.shape)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        print('after fc-layers, the shape is:')
        print(x.shape)
        return x





## Assumption of input data dimension is: [batch_size, C, H, W, Z, seq_len]

## Assumption the dimension for PyTorch model is: [batch_size, seq_len, C, H, W, Z]




class ModelParallelCNN(CNN):
    def __init__(self, devices):
        super(ModelParallelCNN, self).__init__()

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices

        self.seq1 = nn.Sequential(
            self.conv1,
            self.pool,
            self.relu,
            self.drop,
            self.conv2,
            self.pool,
            self.relu,
            self.drop
        ).to(self.devices[0])

        self.seq2 = nn.Sequential(
            self.conv3,
            self.pool,
            self.relu,
            self.drop
        ).to(self.devices[-1])

        #self.fc1 = nn.Linear(40*6*6*6, 1280)   # 4x4x4x80
        self.fc1.to(self.devices[-1])
        self.fc3 = nn.Linear(1280, 1).to(self.devices[-1])
        

        
    def forward(self, x):
        x = x.to(self.devices[0])
        x = self.seq1(x).to(self.devices[-1])
        x = self.seq2(x)
        x = x.view(-1, 40*6*6*6)
        x = self.fc1(x)
        x = self.fc3(x)

        return x




    

class PipelineCNN(ModelParallelCNN):
    def __init__(self, devices, split_size):
        super(PipelineCNN, self).__init__(devices)
        self.split_size = split_size

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices
        
    def forward(self, x):
        
        splits = iter(x.split(self.split_size, dim=0))

        s_next = next(splits)

        s_next = s_next.to(self.devices[0])
        s_prev = self.seq1(s_next).to(self.devices[-1])
        ret = []

        for s_next in splits:
            s_prev = self.seq2(s_prev)
            s_prev = s_prev.view(-1, 40*6*6*6)
            s_prev = self.fc1(s_prev)
            s_prev = self.fc3(s_prev)
            ret.append(s_prev)

            s_next = s_next.to(self.devices[0])
            s_prev = self.seq1(s_next).to(self.devices[-1])

        s_prev = self.seq2(s_prev)
        s_prev = s_prev.view(-1, 40*6*6*6)
        s_prev = self.fc1(s_prev)
        s_prev = self.fc3(s_prev)
        ret.append(s_prev)

        return torch.cat(ret)
    


