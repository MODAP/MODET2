# pylint:disable = not-callable no-member

import torchvision
import torch
from torch import nn
from torch import tensor
import torch.nn.functional as F

# Convolutional Caps Layer
class ConvCapsLayer(nn.Module):
    def __init__(self, length, in_channels, out_channels, kernal_size):

        super(ConvCapsLayer, self).__init__()

        self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernal_size) for _ in length])

    def forward(self, x):
        capsOut = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules] # compile as a capsule, and rezie to [major, random, and 1] dims. for fire detections, there should be 2 classes --- fire, no fire.
        capsOutCat = torch.cat(capsOut, dim=-1)

        return capsOutCat
        
# Class Caps Layer
class ClassCapsLayer(nn.Module):
    def __init__(self, num_classes, routes, in_channels, out_channels, iterations):
        super(ClassCapsLayer, self).__init__()
        
        self.capsules = []
        self.iterations = iterations
        self.route_weights = nn.Parameter(torch.randn(num_classes, routes, in_channels, out_channels)) #https://github.com/gram-ai/capsule-networks/blob/master/capsule_network.py#L63

    @staticmethod
    def squash(s):
        return ((torch.norm(s)**2)/(1+torch.norm(s)**2)) * s/torch.norm(s)
    
    # AcTIve RouTIng!
    def forward(self, x): # x is input
        priorsUij = x @ self.route_weights # We need to pad the dimensions of these two so that they can be dot producted

        logitsBij = torch.Variable(torch.zeros(*priorsUij.size()))# .cuda() uncomment when using cuda
        
        for i in self.iterations:
            probsCi = F.softmax(logitsBij, dim=2) # Calculating softmax probablities
            sumSj = (probsCi*priorsUij).sum(dim=2, keepdim=True) # Calculating probability weighted priors
            outputVj = self.squash(sumSj) # Routing coeffs calculated by squashing the weighted priors
            
            ## Update Bij
            if i != self.iterations-1:
                logitsBij = logitsBij + priorsUij * outputVj

        return outputVj


### Sorry @quantumish. Already implimented active routing... See above.
### We did steal some of your stuff

# class Capsule(nn.Module):
    # def __init__(self, dimensions):
        # self.dims = dimensions
        # self.children = []

    # # We need to initialize connections/weights after all capsules defined
    # def init_connection(self, capsule):
# #        self.weights = [] This will be a problem...
        # self.children.append(capsule)
    
    # # Boilerplate-y for now.
    # def forward(self):
        # self.s = torch.tensor([0.0]*self.dims) # TODO understanding dimensionalities and specifics of calcs will take a bit
        # for child in self.children:
            # # Begin with affine transform of output
            # # child.weights[self] is sketch, trying to indicate $\hat{\vec{u}}_{j|i} = \vec{W}_{ij} \vec{u}_i$
            # # Note that there is no normal u, just the child capsule's s because prev layers output is next layer input
            # uhat = child.weights[self] * child.s
            # # Compute the traditional Wx+b step, in this case $\sum_i c_ij \hat{\vec{u}}_{j|i}$
            # self.s += self.route_consts[child] * uhat
        # # Squashing function
        

