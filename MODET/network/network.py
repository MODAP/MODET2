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
        for i, layer in enumerate(self.capslayers[:-1]): # Creating the connections backwards is more straightforward I guess
            for capsule in layer.capsules:
                for othercapsule in self.capslayers[i+1].capsules:
                    othercapsule.init_connection(capsule)                    
        
    def forward(self, x):
        # FIXME Very questionable
        for capsule in self.capslayers[0].capsules:
            capsule.s = torch.tensor(x)
        for layer in self.capslayers[1:]:
            for capsule in layer.capsules:
                capsule.forward() # Compute all the output values of 


net = CapsNet(512)
net.add_caps_layer(10, 3) # Unsure about how to handle dimensionalities for now
net.add_caps_layer(10, 3)
net.init_connections()
net.forward([1.0,2.0,3.0])
print(net.capslayers[0].capsules[0].s)
