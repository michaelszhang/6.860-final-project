import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self, channels_list, conv_size, pool_size, pool_stride, fc_dim_list, activation=F.relu):
    super().__init__()

    self.conv_layers = nn.ModuleList([nn.Conv2d(channels_list[i], channels_list[i+1], conv_size) for i in range(len(channels_list)-1)])
    self.pool = nn.MaxPool2d(pool_size, pool_stride)
    self.fc_layers = nn.ModuleList([nn.Linear(fc_dim_list[i], fc_dim_list[i+1]) for i in range(len(fc_dim_list)-1)])
    self.activation = activation

  def forward(self, x):
    for conv_layer in self.conv_layers:
      x = self.pool(self.activation(conv_layer(x)))
    
    x = torch.flatten(x, 1)

    for fc_layer in self.fc_layers[:-1]:
      x = self.activation(fc_layer(x))
    
    x = self.fc_layers[-1](x)

    return x