import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalNet(nn.Module):
  def __init__(self, channels_list, conv_size, conv_stride, conv_matrix_dim_list, fc_dim_list, activation=F.relu):
    super().__init__()

    self.channels_list = channels_list
    self.conv_size = conv_size
    self.conv_stride = conv_stride
    self.conv_matrix_dim_list = conv_matrix_dim_list

    self.conv_layers = nn.ModuleList([nn.ModuleList([nn.Conv2d(channels_list[i], channels_list[i+1], conv_size, stride=conv_stride)
                                                     for _ in range(conv_matrix_dim_list[i]**2)])
                                      for i in range(len(channels_list)-1)])
    self.fc_layers = nn.ModuleList([nn.Linear(fc_dim_list[i], fc_dim_list[i+1]) for i in range(len(fc_dim_list)-1)])
    self.activation = activation

  def forward(self, x):
    for i in range(len(self.conv_matrix_dim_list)):
      x_list = [x[:, :, c1:c1+self.conv_size, c2:c2+self.conv_size] for c1 in range(0, self.conv_stride*self.conv_matrix_dim_list[i], self.conv_stride)
                                                                    for c2 in range(0, self.conv_stride*self.conv_matrix_dim_list[i], self.conv_stride)]
      x_transformed_list = [self.activation(self.conv_layers[i][j](x_list[j])) for j in range(self.conv_matrix_dim_list[i]**2)]
      x = torch.cat(x_transformed_list, 2)
      x = torch.reshape(x, (-1, self.channels_list[i+1], self.conv_matrix_dim_list[i], self.conv_matrix_dim_list[i]))
    
    x = torch.flatten(x, 1)

    for fc_layer in self.fc_layers[:-1]:
      x = self.activation(fc_layer(x))
    
    x = self.fc_layers[-1](x)

    return x