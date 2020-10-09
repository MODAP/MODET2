# pylint:disable = not-callable no-member

import torchvision
import torch
from torch import nn
from torch import tensor
import torch.nn.functional as F

# Convolutional Caps Layer
class ConvCapsLayer(nn.Module):
    def __init__(self, length, in_channels, out_channels, kernal_size):
        self.capsules = [nn.Conv2d(in_channels, out_channels, kernal_size) for i in length]

# Class Caps Layer
class ClassCapsLayer(nn.Module):
    def __init__(self):
        self.capsules = []

    
class Capsule(nn.Module):
    def __init__(self, dimensions):
        self.dim = dimensions
        self.children = []

    # We need to initialize connections/weights after all capsules defined
    def init_connection(self, capsule):
        self.children.append(capsule)
    
    # Boilerplate-y for now.
    def forward(self, x):
        self.s = torch.tensor([0]*self.dims) # TODO understanding dimensionalities and specifics of calcs will take a bit
        for child in self.children:
            # Begin with affine transform of output
            # child.weights[self] is sketch, trying to indicate $\hat{\vec{u}}_{j|i} = \vec{W}_{ij} \vec{u}_i$
            # Note that there is no normal u, just the child capsule's s because prev layers output is next layer input
            uhat = child.weights[self] * child.s
            # Compute the traditional Wx+b step, in this case $\sum_i c_ij \hat{\vec{u}}_{j|i}$
            self.s += self.route_consts[child] * uhat
        # Squashing function
        self.s = ((torch.norm(self.s)**2)/(1+torch.norm(self.s)**2)) * self.s/torch.norm(self.s)

# Pulling a Zoch
if __name__ == '__main__':
    cap = Capsule(15)


