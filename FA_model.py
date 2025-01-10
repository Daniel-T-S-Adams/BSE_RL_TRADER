import torch
from torch import nn, Tensor
import torch.optim as optim
from typing import Dict, DefaultDict, Tuple, Iterable

from config.config_params import CONFIG

class NeuralNet(nn.Module):
    """
    A fully connected neural network class with 
    specified dimensions and output activation.

    Attributes:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        network (nn.Module): The sequential neural network created from the specified dimensions.
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None):
        """
        Initializes the Network with the given dimensions and output activation.

        Args:
            dims (Iterable[int]): A list or tuple of integers specifying the sizes of each layer in the network.
                                  The first element is the input size, and the last element is the output size.
            output_activation (nn.Module): The activation function to apply to the output layer. Default is None.
        """
        super().__init__()
        self.input_size = dims[0]
        self.output_size = dims[-1]
        self.network = self.make_seq(dims, output_activation)

    
    def make_seq(self, dims: Iterable[int], output_activation: nn.Module) -> nn.Module:
        """
        Creates a sequential neural network from the specified dimensions and output activation.

        Args:
            dims (Iterable[int]): A list or tuple of integers specifying the sizes of each layer in the network.
            output_activation (nn.Module): The activation function to apply to the output layer.

        Returns:
            nn.Module: A sequential neural network constructed from the specified layers and activation functions.
        """
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.5))
        if output_activation:
            layers.append(output_activation)
        return nn.Sequential(*layers)
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes a forward pass through the network.

        Args:
            x: The input tensor to the network.

        Returns:
            The output of the network after passing through the sequential layers.
        """
        return self.network(x)

def train_network(model, optimizer, criterion, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()



def normalize_data_min_max(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Normalize inputs and targets using Min-Max normalization. If Max = Min it replaces max - min with epsilon

    Parameters:
        inputs (torch.Tensor): Input tensor of shape (batch_size, input_size).
        targets (torch.Tensor): Target tensor of shape (batch_size,).

    Returns:
        inputs_normalized (torch.Tensor): Normalized input tensor.
        targets_normalized (torch.Tensor): Normalized target tensor.
        normalization_params (Dict): Min and max values for reversing normalization.
    """
    # Normalize inputs
    x_min, _ = torch.min(inputs, dim=0)
    x_max, _ = torch.max(inputs, dim=0)
    x_range = x_max - x_min  
    epsilon = 1e-4
    x_range[x_range == 0] = epsilon  # if this takes too long we could try just adding epsilon to the range.
    inputs_normalized = (inputs - x_min) / (x_range)
    
    # Normalize targets
    y_min = torch.min(targets)
    y_max = torch.max(targets)
    y_range = y_max - y_min
    y_range[y_range == 0] = epsilon
    targets_normalized = (targets - y_min) / (y_range)

    # Save normalization parameters for reversing
    normalization_params = {
        "x_min": x_min,
        "x_range": x_range,
        "y_min": y_min,
        "y_range": y_range,
    }

    return inputs_normalized, targets_normalized, normalization_params
