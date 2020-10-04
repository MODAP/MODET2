import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from capsule import *

class CapsNet(nn.Module):
    def __init__(self, input_size):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(input_size, 256, 4)
        self.capslayers = []

    def add_caps_layer(self, capsules, dims):
        self.capslayers.append(CapsLayer(capsules, dims))
        
    def init_connections(self):
        for layer in reversed(self.capslayers[1:]): # Creating the connections backwards is more straightforward I guess
            for i, capsule in layer.capsules:
                capsule.init_connection(layer.capsules[i+1]) 
        
    def forward(self, x):
        for layer in self.capslayers:
            for capsule in layer.capsules:
                capsule.forward() # Compute all the output values of 


net = CapsNet(512)
net.add_caps_layer(10, 3) # Unsure about how to handle dimensionalities for now
net.init_connections()
net.forward([0.0,0.0,0.0])
