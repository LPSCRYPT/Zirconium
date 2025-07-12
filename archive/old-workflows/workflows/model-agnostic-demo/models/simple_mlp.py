#!/usr/bin/env python3
"""
Simple MLP Model Implementation

A basic multi-layer perceptron for testing the model-agnostic pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
from ezkl_pipeline import ModelInterface

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 20, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

class SimpleMLP_Model(ModelInterface):
    """Simple MLP implementing the ModelInterface"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 20, output_size: int = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create model instance
        self._model = SimpleMLP(input_size, hidden_size, output_size)
        self._model.eval()
    
    def get_model(self) -> torch.nn.Module:
        """Return the PyTorch model instance"""
        return self._model
    
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input tensor for the model"""
        # Simple 4-dimensional input (like Iris dataset)
        return torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)
    
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL JSON format"""
        return [[5.1, 3.5, 1.4, 0.2]]
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            "name": "simple_mlp",
            "description": "Simple Multi-Layer Perceptron for classification",
            "architecture": "3-layer MLP with ReLU activation",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "domain": "classification",
            "proven_working": False,
            "gas_estimate": "TBD",
            "proof_size_kb": "TBD"
        }