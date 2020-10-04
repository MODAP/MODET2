import torchvision
import torch
from torch import nn
import torch.nn.functional as F

# Let's enable this when we need to.
# import sys
# sys.setrecursionlimit(15000)

class CapsNet(nn.Module):
    def __init__(self, input_size):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(input_size, 256, 4)
        self.capslayers = []

    def init_connections(self):
        for layer in self.capslayers[:-1]:
            for i, capsule in layer.capsules:
                capsule.init_connection(layer.capsules[i+1]) 
        
    def forward(self, x):
        pass 

# Maybe have CapsLayer class? TODO Think about if actually needed
class CapsLayer(nn.Module):
    def __init__(self):
        self.capsules = []
        self.childCapsules = []
    
class Capsule(nn.Module):
    def __init__(self, dimensions):
        self.dim = dimensions
        pass

    # We need to initialize connections/weights after all capsules defined
    def init_connection(self, capsule):
        self.childCapsules.append(capsule)
    
    # Boilerplate-y for now.
    def forward(self, x):
        self.s = torch.tensor([0]*self.dim) # TODO understanding dimensionalities and specifics of calcs will take a bit TODO torch.tensor is not callable?
        for child in self.childCapsules:
            # Begin with affine transform of output
            # child.weights[self] is sketch, trying to indicate $\hat{\vec{u}}_{j|i} = \vec{W}_{ij} \vec{u}_i$
            uhat = child.weights[self] * child.u
            # Compute the traditional Wx+b step, in this case $\sum_i c_ij \hat{\vec{u}}_{j|i}$
            self.s += self.route_consts[child] * uhat
        # Squashing function
        self.s = ((torch.norm(self.s)**2)/(1+torch.norm(self.s)**2)) * self.s/torch.norm(self.s)

