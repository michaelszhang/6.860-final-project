import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
  def __init__(self, tree_depth, activation):
    super().__init__()
    self.activation = activation
    self.layers = nn.ModuleList([nn.Linear(2**(tree_depth-i), 2**(tree_depth-i-1), bias=False) for i in range(tree_depth)])
  
  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.activation(layer(x))
    
    x = self.layers[-1](x)
    
    return x