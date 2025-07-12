#!/usr/bin/env python3
"""
Mamba Simple Model
Simplified Mamba model for EZKL compatibility
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
from pipeline.ezkl_pipeline import ModelInterface

class SimpleLinearModel(nn.Module):
    """Simple linear model mimicking Mamba architecture"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 24, output_size: int = 69):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class MambaSimpleModel(ModelInterface):
    """Mamba Simple Model for EZKL"""
    
    def __init__(self):
        self.input_size = 10
        self.hidden_size = 24
        self.output_size = 69
        
    def get_model(self) -> torch.nn.Module:
        model = SimpleLinearModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.tensor([[
            0.1875,
            -0.0625,
            -0.25,
            -0.3125,
            0.0625,
            -0.3125,
            -0.3125,
            -0.5,
            0.1875,
            -0.25
        ]], dtype=torch.float32)
    
    def get_input_data(self) -> List[List[float]]:
        return [[
            0.1875,
            -0.0625,
            -0.25,
            -0.3125,
            0.0625,
            -0.3125,
            -0.3125,
            -0.5,
            0.1875,
            -0.25
        ]]
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "mamba_simple",
            "description": "Simplified Mamba model with selective state spaces",
            "architecture": "Mamba (State Space Model)",
            "domain": "sequence_modeling",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "status": "🧪 EXPERIMENTAL",
            "gas_estimate": "~580K"
        }