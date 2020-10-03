import torchvision
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.setrecursionlimit(15000)

class CapsNet(nn.Module):
    def __init__(self, input_size):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(input_size, 256, 4)
        
    def forward(self, x):
        pass 

class Capsule(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

