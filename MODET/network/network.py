import torchvision
import torch
from torch import nn
import torch.nn.functional as F

class CapsNet(nn.Module):
    def __init__(self, input_size):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(input_size, 256, 4)
        self.capslayers = []

    def init_connections():
        for layer in capslayers[:-1]:
            for i, capsule in layer.capsules:
                capsule.init_connection(layer.capsules[i+1]) 
        
    def forward(self, x):
        pass 


