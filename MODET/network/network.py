# pylint:disable = import-error
# pylint:disable = import-error

import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from capsule import ConvCapsLayer, ClassCapsLayer

class CapsNet(nn.Module):
    def __init__(self, input_size):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(input_size, 256, 4) # TODO please add input annotations
        self.primary_conv_capsule = ConvCapsLayer(1, 256, 256, 4) # arbitrary numbers FIXME and also add variable annotations
        self.digi_class_caps = ClassCapsLayer(1, 256, 256, 4, 10) # More aribitrary numbers FIXME


    def forward(self, x):
        x = F.relu(self.conv(x), inplace = True) # Taking input and feeding it through conv and using ReLu
        x = self.primary_conv_capsule(x)
        x = Digi_class_caps(x).squeeze().transpose(0, 1) # TODO strange dimensional stuff that was done in the example, but may not be necessary?
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        return classes

if __name__ == '__main__':
    pass
#    net = CapsNet(512)
#    net.add_caps_layer(10, 3) Unsure about how to handle dimensionalities for now
#    net.add_caps_layer(10, 3)
#    net.init_connections()
#    net.forward([1.0,2.0,3.0])
#    print(net.capslayers[0].capsules[0].s)
