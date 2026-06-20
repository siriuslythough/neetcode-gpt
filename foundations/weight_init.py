import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Set seed once at the start
        torch.manual_seed(0)
        
        # Track dimensions across all layers
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # 1. First, generate ALL weight matrices to match the expected RNG order
        for i in range(num_layers):
            if init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            else:
                std = 1.0
                
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)
            
        # 2. Next, generate the random input tensor
        x = torch.randn(1, input_dim)
        stds = []
        
        # 3. Finally, execute the sequential forward passes
        for w in weights:
            x = x @ w.t()
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
            
        return stds
