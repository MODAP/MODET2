# pylint:disable = import-error
# pylint:disable = import-error

import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from .capsule import ConvCapsLayer, ClassCapsLayer

class CapsNet(nn.Module):
    def __init__(self, input_channels, number_classes, initial_output_channels=256, capsule_output_channels=4, number_capsules=16, kernel_size=4, routing_iterations=8):
        super(CapsNet, self).__init__()

        self.conv = nn.Conv2d(input_channels, initial_output_channels, kernel_size) # TODO please add input annotations
        self.primary_conv_capsule = ConvCapsLayer(number_capsules, initial_output_channels, initial_output_channels, kernel_size) # arbitrary numbers FIXME and also add variable annotations
        self.digi_class_caps = ClassCapsLayer(number_classes, number_capsules, number_capsules,  capsule_output_channels, routing_iterations) # More aribitrary numbers FIXME


    def forward(self, x):
        x = F.relu(self.conv(x), inplace = True) # Taking input and feeding it through conv and using ReLu
        x = self.primary_conv_capsule(x)
        x = self.digi_class_caps(x).squeeze().transpose(0, 1) # TODO strange dimensional stuff that was done in the example, but may not be necessary?
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        return classes

if __name__ == '__main__':
    pass

