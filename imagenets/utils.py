import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

def get_balanced_subset(dataset, num_classes, examples_per_class):
  targets = dataset.targets
  if not torch.is_tensor(targets):
    targets = torch.Tensor(targets)
  all_idxs = torch.arange(len(dataset))
  subset_idx_list = []

  for i in range(num_classes):
    class_idxs = all_idxs[targets[all_idxs] == i]
    perm = torch.randperm(len(class_idxs))[:examples_per_class]
    subset_class_idxs = class_idxs[perm]
    subset_idx_list.append(subset_class_idxs)
  
  subset_idxs = torch.cat(subset_idx_list)

  return Subset(dataset, subset_idxs)

def train_model(trainloader, model, criterion, optimizer, num_epochs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for _ in range(num_epochs):
    for inputs, labels in trainloader:
      inputs.to(device)
      labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  
  model.cpu()

def test_model(testloader, model):
  total, correct = 0, 0

  with torch.no_grad():
    for inputs, labels in testloader:
      outputs = model(inputs)
      predictions = torch.argmax(outputs, 1)
      total += labels.size(0)
      correct += (predictions == labels).sum().item()
  
  return total, correct

def fgsm_examples(inputs, labels, model, criterion, eps):
  inputs_rg = inputs.clone().detach().requires_grad_(True)
  outputs = model(inputs_rg)
  loss = criterion(outputs, labels)
  loss.backward()
  deltas = torch.sign(inputs_rg.grad) * eps

  return torch.clamp(inputs + deltas, -1, 1)

def test_model_fgsm(testloader, model, criterion, eps):
  total, correct = 0, 0
  original_inputs = None
  perturbed_inputs = None

  for inputs, labels in testloader:
    fgsm_inputs = fgsm_examples(inputs, labels, model, criterion, eps)
    outputs = model(fgsm_inputs)
    predictions = torch.argmax(outputs, 1)
    total += labels.size(0)
    correct += (predictions == labels).sum().item()

    if original_inputs is None:
      original_inputs = inputs
    if perturbed_inputs is None:
      perturbed_inputs = fgsm_inputs
  
  return total, correct, original_inputs, perturbed_inputs