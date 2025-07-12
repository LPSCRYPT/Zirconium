#!/usr/bin/env python3
"""
Simple Chain Models

Composable versions of existing MAP models, adapted for proof composition.
These models implement ComposableModelInterface for seamless chaining.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add interfaces to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'interfaces'))
from composable_model import ComposableModelInterface

class ComposableFeatureExtractor(ComposableModelInterface):
    """
    Feature extraction model - first stage in chain.
    Takes raw input and produces feature representations.
    """
    
    def __init__(self):
        self.input_size = 10
        self.output_size = 8
        
    def get_model(self) -> torch.nn.Module:
        model = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            nn.ReLU()
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, self.input_size)
    
    def get_input_data(self) -> List[List[float]]:
        """Raw sensor data"""
        return [[1.5, -0.8, 2.1, 0.3, -1.2, 0.9, -0.4, 1.8, 0.6, -0.7]]
    
    def get_input_shape(self) -> List[int]:
        return [self.input_size]
    
    def get_output_shape(self) -> List[int]:
        return [self.output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """This is the first model, so input comes from external source"""
        return data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """Normalize features for next model"""
        # Simple normalization
        normalized_data = []
        for sample in data:
            # Normalize to [-1, 1] range
            max_val = max(abs(x) for x in sample)
            if max_val > 0:
                normalized_sample = [x / max_val for x in sample]
            else:
                normalized_sample = sample
            normalized_data.append(normalized_sample)
        return normalized_data
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "feature_extractor",
            "description": "Feature extraction from raw input",
            "architecture": "2-layer MLP with ReLU",
            "domain": "feature_extraction",
            "status": "🔗 COMPOSABLE",
            "chain_position": "input",
            "input_size": self.input_size,
            "output_size": self.output_size
        }

class ComposableClassifier(ComposableModelInterface):
    """
    Classification model - middle stage in chain.
    Takes features and produces class probabilities.
    """
    
    def __init__(self):
        self.input_size = 8
        self.output_size = 5
        
    def get_model(self) -> torch.nn.Module:
        model = nn.Sequential(
            nn.Linear(self.input_size, 12),
            nn.ReLU(),
            nn.Linear(12, self.output_size),
            nn.Softmax(dim=1)
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, self.input_size)
    
    def get_input_data(self) -> List[List[float]]:
        """This will be overridden by transform_input from previous model"""
        return [[0.1, -0.2, 0.3, 0.0, -0.1, 0.4, -0.3, 0.2]]
    
    def get_input_shape(self) -> List[int]:
        return [self.input_size]
    
    def get_output_shape(self) -> List[int]:
        return [self.output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """Accept normalized features from feature extractor"""
        # Ensure input has correct size
        transformed_data = []
        for sample in data:
            if len(sample) != self.input_size:
                # Pad or truncate to match expected input size
                if len(sample) > self.input_size:
                    transformed_sample = sample[:self.input_size]
                else:
                    transformed_sample = sample + [0.0] * (self.input_size - len(sample))
            else:
                transformed_sample = sample
            transformed_data.append(transformed_sample)
        return transformed_data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """Pass probabilities to next model"""
        return data
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "classifier",
            "description": "Classification from extracted features",
            "architecture": "2-layer MLP with Softmax",
            "domain": "classification",
            "status": "🔗 COMPOSABLE",
            "chain_position": "middle",
            "input_size": self.input_size,
            "output_size": self.output_size
        }

class ComposableDecisionMaker(ComposableModelInterface):
    """
    Decision making model - final stage in chain.
    Takes class probabilities and produces final decisions.
    """
    
    def __init__(self):
        self.input_size = 5
        self.output_size = 3
        
    def get_model(self) -> torch.nn.Module:
        model = nn.Sequential(
            nn.Linear(self.input_size, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_size),
            nn.Sigmoid()  # Output decisions as confidence scores
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, self.input_size)
    
    def get_input_data(self) -> List[List[float]]:
        """This will be overridden by transform_input from previous model"""
        return [[0.2, 0.3, 0.1, 0.3, 0.1]]  # Probability distribution
    
    def get_input_shape(self) -> List[int]:
        return [self.input_size]
    
    def get_output_shape(self) -> List[int]:
        return [self.output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """Accept probabilities from classifier"""
        # Ensure probabilities sum to 1 (normalize if needed)
        transformed_data = []
        for sample in data:
            sample_sum = sum(sample)
            if sample_sum > 0:
                normalized_sample = [x / sample_sum for x in sample]
            else:
                normalized_sample = [1.0 / len(sample)] * len(sample)
            transformed_data.append(normalized_sample)
        return transformed_data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """Final output - convert to decisions"""
        # Convert confidence scores to binary decisions (threshold = 0.5)
        decision_data = []
        for sample in data:
            decisions = [1.0 if x > 0.5 else 0.0 for x in sample]
            decision_data.append(decisions)
        return decision_data
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "decision_maker",
            "description": "Final decision making from classification probabilities",
            "architecture": "2-layer MLP with Sigmoid",
            "domain": "decision_making",
            "status": "🔗 COMPOSABLE",
            "chain_position": "output",
            "input_size": self.input_size,
            "output_size": self.output_size
        }

class ComposableWeatherProcessor(ComposableModelInterface):
    """
    Weather data processing model - specialized for weather chains.
    Can be used as input or middle stage.
    """
    
    def __init__(self):
        self.input_size = 10  # Weather features
        self.output_size = 6   # Processed weather features
        
    def get_model(self) -> torch.nn.Module:
        # Simplified version of weather xLSTM for composition
        model = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.Tanh(),  # Weather data benefits from Tanh
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.output_size)
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, self.input_size)
    
    def get_input_data(self) -> List[List[float]]:
        """Sample weather data: temp, humidity, pressure, wind_speed, etc."""
        return [[22.5, 65.0, 1013.2, 12.3, 0.8, 15.6, 8.9, 45.2, 0.1, 280.5]]
    
    def get_input_shape(self) -> List[int]:
        return [self.input_size]
    
    def get_output_shape(self) -> List[int]:
        return [self.output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """Normalize weather data to standard ranges"""
        transformed_data = []
        for sample in data:
            # Simple normalization for weather data
            # In practice, this would use domain-specific scaling
            normalized_sample = []
            for i, value in enumerate(sample):
                # Apply different normalization based on weather parameter type
                if i == 0:  # Temperature (assume Celsius, scale to [-1,1])
                    normalized_value = (value - 20) / 50  # Rough scaling
                elif i == 1:  # Humidity (0-100%)
                    normalized_value = (value - 50) / 50
                elif i == 2:  # Pressure (around 1013 hPa)
                    normalized_value = (value - 1013) / 50
                else:  # Other parameters
                    normalized_value = value / 100  # Generic scaling
                normalized_sample.append(max(-1, min(1, normalized_value)))
            transformed_data.append(normalized_sample)
        return transformed_data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """Scale processed weather features for next model"""
        return data
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "weather_processor",
            "description": "Weather data processing and normalization",
            "architecture": "3-layer MLP with Tanh/ReLU",
            "domain": "weather_processing",
            "status": "🔗 COMPOSABLE",
            "chain_position": "input_or_middle",
            "input_size": self.input_size,
            "output_size": self.output_size
        }

class ComposableSequenceModel(ComposableModelInterface):
    """
    Sequence processing model - for temporal data chains.
    Uses simplified RNN architecture.
    """
    
    def __init__(self):
        self.input_size = 6
        self.hidden_size = 8
        self.output_size = 4
        
    def get_model(self) -> torch.nn.Module:
        # Note: This is simplified for EZKL compatibility
        # Real sequence models would use RNN layers
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, self.input_size)
    
    def get_input_data(self) -> List[List[float]]:
        """This will be overridden by previous model in chain"""
        return [[0.1, 0.2, -0.1, 0.3, -0.2, 0.4]]
    
    def get_input_shape(self) -> List[int]:
        return [self.input_size]
    
    def get_output_shape(self) -> List[int]:
        return [self.output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """Accept processed features from previous model"""
        return data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """Apply temporal smoothing to outputs"""
        # Simple smoothing - in practice this would be more sophisticated
        smoothed_data = []
        for sample in data:
            # Apply mild smoothing by scaling
            smoothed_sample = [x * 0.9 for x in sample]
            smoothed_data.append(smoothed_sample)
        return smoothed_data
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "sequence_processor",
            "description": "Sequence processing with temporal smoothing",
            "architecture": "3-layer MLP (RNN-like)",
            "domain": "sequence_processing",
            "status": "🔗 COMPOSABLE",
            "chain_position": "middle_or_output",
            "input_size": self.input_size,
            "output_size": self.output_size
        }