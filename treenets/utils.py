import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm

class TreeData():
  def __init__(self, tree_depth, activation):
    self.tree_depth = tree_depth
    self.activation = activation
    self.weight_list = [torch.rand(2, 1)*2 for _ in range(tree_depth)]

  def calc_output(self, x, noise_eps):
    for i in range(self.tree_depth-1):
      x_list = [x[:, c:c+2] for c in range(0, 2**(self.tree_depth-i), 2)]
      x_next = [self.activation(torch.matmul(x, self.weight_list[i])) for x in x_list]
      x = torch.cat(x_next, 1)
    
    x = torch.matmul(x, self.weight_list[-1])+torch.normal(torch.zeros(1), noise_eps)
    
    return x

  def generate_data(self, num_examples, noise_eps=0):
    x = torch.normal(torch.zeros(num_examples, 2**self.tree_depth), torch.ones(num_examples, 2**self.tree_depth))
    y = self.calc_output(x, noise_eps)

    return [(x[i], y[i]) for i in range(num_examples)]

def train_model(trainloader, model, criterion, optimizer, scheduler, num_epochs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for _ in range(num_epochs):
    for x, y in trainloader:
      x.to(device)
      y.to(device)

      optimizer.zero_grad()
      y_hat = model(x)
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
    
    scheduler.step()
  
  model.cpu()

def test_model(testloader, model, criterion):
  total_loss = 0

  with torch.no_grad():
    for x, y in testloader:
      y_hat = model(x)
      total_loss += criterion(y_hat, y)
  
  average_loss = total_loss / len(testloader)
  
  return average_loss