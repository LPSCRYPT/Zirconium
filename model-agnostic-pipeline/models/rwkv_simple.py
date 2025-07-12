#!/usr/bin/env python3
"""
RWKV Simple Model
Simplified RWKV model for EZKL compatibility
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
from ezkl_pipeline import ModelInterface

class SimpleLinearModel(nn.Module):
    """Simple linear model mimicking RWKV architecture"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 16, output_size: int = 69):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class RWKVSimpleModel(ModelInterface):
    """RWKV Simple Model for EZKL"""
    
    def __init__(self):
        self.input_size = 10
        self.hidden_size = 16
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
            0.4652051031589508,
            -0.9299561381340027,
            -1.242055892944336,
            1.0063905715942383,
            1.0123469829559326,
            -0.12968644499778748,
            -0.7979132533073425,
            -0.5822405815124512,
            2.378202199935913,
            1.0034712553024292
        ]], dtype=torch.float32)
    
    def get_input_data(self) -> List[List[float]]:
        return [[
            0.4652051031589508,
            -0.9299561381340027,
            -1.242055892944336,
            1.0063905715942383,
            1.0123469829559326,
            -0.12968644499778748,
            -0.7979132533073425,
            -0.5822405815124512,
            2.378202199935913,
            1.0034712553024292
        ]]
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "rwkv_simple",
            "description": "Simplified RWKV model for language modeling",
            "architecture": "RWKV (Receptance Weighted Key Value)",
            "domain": "sequence_modeling",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "proven_working": True,
            "gas_estimate": "614K"
        }