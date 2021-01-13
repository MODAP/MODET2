import torch

from MODET.data import preprocessing
from MODET.network import network


capsNet = network.CapsNet(3, 10)


# ds = preprocessing.Dataset("./export-2020-10-30T16_41_24.239Z.csv")
david = torch.rand((64, 3, 32, 32))

print(capsNet(david))
