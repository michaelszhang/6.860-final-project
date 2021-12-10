import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
  def __init__(self, dim_list, activation=F.relu):
    super().__init__()

    self.fc_layers = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list)-1)])
    self.activation = activation

  def forward(self, x):
    x = torch.flatten(x, 1)

    for fc_layer in self.fc_layers[:-1]:
      x = self.activation(fc_layer(x))
    
    x = self.fc_layers[-1](x)

    return x