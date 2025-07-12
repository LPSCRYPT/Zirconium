#!/usr/bin/env python3
"""
Model Registry

Factory pattern for easy model swapping and registration.
"""

from typing import Dict, Type, List
import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
from ezkl_pipeline import ModelInterface

# Import available models
from weather_xlstm import WeatherxLSTM
from simple_mlp import SimpleMLP_Model
from rwkv_simple import RWKVSimpleModel
from mamba_simple import MambaSimpleModel
from xlstm_simple import XLSTMSimpleModel

class ModelRegistry:
    """Registry for managing available models"""
    
    def __init__(self):
        self._models: Dict[str, Type[ModelInterface]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register the built-in models"""
        self.register("weather_xlstm", WeatherxLSTM)
        self.register("simple_mlp", SimpleMLP_Model)
        self.register("rwkv_simple", RWKVSimpleModel)
        self.register("mamba_simple", MambaSimpleModel)
        self.register("xlstm_simple", XLSTMSimpleModel)
    
    def register(self, name: str, model_class: Type[ModelInterface]):
        """Register a new model class"""
        self._models[name] = model_class
        print(f"📝 Registered model: {name}")
    
    def create_model(self, name: str, **kwargs) -> ModelInterface:
        """Create a model instance by name"""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        
        model_class = self._models[name]
        return model_class(**kwargs)
    
    def list_models(self) -> List[str]:
        """List all available model names"""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict:
        """Get information about a specific model"""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        # Create a temporary instance to get config
        model = self.create_model(name)
        return model.get_config()
    
    def print_available_models(self):
        """Print information about all available models"""
        print("🔧 Available Models:")
        print("-" * 50)
        
        for name in self._models.keys():
            try:
                config = self.get_model_info(name)
                status = "✅ PROVEN" if config.get("proven_working") else "🧪 EXPERIMENTAL"
                print(f"   {name}: {config.get('description', 'No description')} {status}")
                print(f"      Architecture: {config.get('architecture', 'Unknown')}")
                print(f"      Domain: {config.get('domain', 'Unknown')}")
                if config.get("gas_estimate"):
                    print(f"      Gas estimate: {config.get('gas_estimate')}")
                print()
            except Exception as e:
                print(f"   {name}: Error getting info - {e}")

# Global registry instance
registry = ModelRegistry()

def get_model(name: str, **kwargs) -> ModelInterface:
    """Convenience function to get a model by name"""
    return registry.create_model(name, **kwargs)

def list_available_models() -> List[str]:
    """Convenience function to list available models"""
    return registry.list_models()

def print_models():
    """Convenience function to print model information"""
    registry.print_available_models()