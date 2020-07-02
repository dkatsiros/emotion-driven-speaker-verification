import copy
import re
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


class CNN2(nn.Module):
    def __init__(self, output_dim=7, regression=False, multitask=False):
        super(CNN2, self).__init__()

        self.regression = regression
        self.multitask = multitask

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)  # 2d convolution
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=3)
        self.conv5 = nn.Conv2d(80, 160, kernel_size=3)

        self.conv1_bn = nn.BatchNorm2d(10)  # batch normalization
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5_bn = nn.BatchNorm2d(160)

        self.dropout1 = nn.Dropout2d(0.25)  # dropout
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.1)

        # self.fc0 = nn.Linear(18480, 7200)
        self.fc1 = nn.Linear(1920, 1024)   # linear layer 7200
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        # input: (batch_size,1,max_seq,features)
        # Each layer applies the following matrix tranformation
        # recursively: (batch_size,conv_output,max_seq/2 -1,features/2 -1)

        x = F.max_pool2d(F.leaky_relu(self.conv1_bn(self.conv1(x))),
                         2)  # [14, 10, 119, 63]
        x = F.max_pool2d(F.leaky_relu(self.conv2_bn(self.conv2(x))),
                         2)  # [14, 20, 58, 30]

        x = self.dropout1(x)  # [14, 20, 58, 30]
        x = F.max_pool2d(F.leaky_relu(self.conv3_bn(self.conv3(x))),
                         2)  # [14, 40, 28, 14]

        x = F.max_pool2d(F.leaky_relu(self.conv4_bn(self.conv4(x))),
                         2)  # [14, 80, 13, 6]

        x = F.max_pool2d(F.leaky_relu(self.conv5_bn(self.conv5(x))),
                         2)  # [14,160,5,2]

        # batch_size x new_dim(ginomeno ton allon)
        x = self.dropout2(F.leaky_relu(self.fc1(x.view(len(x), - 1))))
        x = self.dropout2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        if self.multitask:
            pass
        elif self.regression:
            x = x.view(-1)
        else:
            # x = F.log_softmax(x)  # , dim=1)
            return x

        return x

    @ staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
