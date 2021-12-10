import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalNet(nn.Module):
  def __init__(self, tree_depth, activation):
    super().__init__()
    self.tree_depth = tree_depth
    self.activation = activation
    self.layers = nn.ModuleList([nn.ModuleList([nn.Linear(2, 1, bias=False) for _ in range(2**(self.tree_depth-i-1))]) for i in range(tree_depth)])
  
  def forward(self, x):
    for i in range(self.tree_depth-1):
      x_list = [x[:, c:c+2] for c in range(0, 2**(self.tree_depth-i), 2)]
      x_transformed_list = [self.activation(self.layers[i][j](x_list[j])) for j in range(2**(self.tree_depth-i-1))]
      x = torch.cat(x_transformed_list, dim=1)
    
    x = self.layers[-1][0](x)
    
    return x