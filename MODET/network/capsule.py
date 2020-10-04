import torchvision
import torch
from torch import nn
import torch.nn.functional as F

# Maybe have CapsLayer class? TODO Think about if actually needed
class CapsLayer(nn.Module):
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
