import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNet(nn.Module):
  def __init__(self, channels_list, conv_size_list, padding_list, fc_dim_list, activation=F.relu):
    super().__init__()

    self.conv_layers = nn.ModuleList([nn.Conv2d(channels_list[i], channels_list[i+1], conv_size_list[i], padding=padding_list[i]) for i in range(len(channels_list)-1)])
    self.fc_layers = nn.ModuleList([nn.Linear(fc_dim_list[i], fc_dim_list[i+1]) for i in range(len(fc_dim_list)-1)])
    self.activation = activation

  def forward(self, x):
    for conv_layer in self.conv_layers:
      x = self.activation(conv_layer(x))
    
    x = torch.flatten(x, 1)

    for fc_layer in self.fc_layers[:-1]:
      x = self.activation(fc_layer(x))
    
    x = self.fc_layers[-1](x)

    return x