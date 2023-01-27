# -*- coding: utf-8 -*-
"""

Author Shaheim Ogbomo-Harmitt

CNN - Pytorch Code 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()

        # Convolutional Layers
        
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.conv2 = nn.Conv2d(32, 32, 2)
        self.conv3 = nn.Conv2d(32, 32, 2)
        self.conv4 = nn.Conv2d(32, 32, 2)
        
    
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.9)
        self.bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(2048, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        
        #self.fc1 = nn.Linear(512, 256)  
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        
        x = self.relu1((self.conv1(x)))
        x = self.maxpool1(x)
        
        x = self.relu2((self.conv2(x)))
        x = self.maxpool2(x)
        
        x = self.relu3((self.conv3(x)))
        x = self.maxpool3(x)
        
        x = self.relu4((self.conv4(x)))
        x = self.maxpool4(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop(x)
        
        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        
        return x