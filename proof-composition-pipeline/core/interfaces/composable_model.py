#!/usr/bin/env python3
"""
Composable Model Interface

Extended interface for models that can be chained together in proof composition.
Builds on the ModelInterface from MAP but adds composition capabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import sys
import os

# Add MAP core to path to reuse ModelInterface
map_core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model-agnostic-pipeline', 'core')
sys.path.append(map_core_path)
from ezkl_pipeline import ModelInterface

class ComposableModelInterface(ModelInterface):
    """
    Extended interface for models that can be composed in chains.
    
    Adds composition-specific methods to the base ModelInterface:
    - Input/output shape compatibility checking
    - Data transformation between models
    - Chain position metadata
    """
    
    @abstractmethod
    def get_output_shape(self) -> List[int]:
        """Return the output tensor shape (without batch dimension)"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> List[int]:
        """Return the input tensor shape (without batch dimension)"""
        pass
    
    @abstractmethod
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        """
        Transform incoming data to this model's expected format.
        Used when this model receives output from a previous model in the chain.
        
        Args:
            data: Output data from previous model in chain
            
        Returns:
            Transformed data compatible with this model's input format
        """
        pass
    
    @abstractmethod
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        """
        Transform this model's output for the next model in the chain.
        
        Args:
            data: Raw output from this model
            
        Returns:
            Transformed data compatible with next model's input format
        """
        pass
    
    def is_compatible_with(self, next_model: 'ComposableModelInterface') -> bool:
        """
        Check if this model's output is compatible with the next model's input.
        
        Args:
            next_model: The next model in the composition chain
            
        Returns:
            True if models are compatible, False otherwise
        """
        try:
            my_output_shape = self.get_output_shape()
            next_input_shape = next_model.get_input_shape()
            
            # Basic shape compatibility check
            # More sophisticated checks can be implemented by subclasses
            return len(my_output_shape) == len(next_input_shape)
        except Exception:
            return False
    
    def get_composition_metadata(self) -> Dict[str, Any]:
        """
        Get metadata specific to model composition.
        
        Returns:
            Dictionary with composition-relevant metadata
        """
        base_config = self.get_config()
        
        composition_metadata = {
            **base_config,
            "input_shape": self.get_input_shape(),
            "output_shape": self.get_output_shape(),
            "composition_compatible": True,
            "supports_chaining": True
        }
        
        return composition_metadata

class ChainPosition:
    """Represents a model's position and role in a composition chain"""
    
    def __init__(self, 
                 model: ComposableModelInterface,
                 position: int,
                 chain_id: str,
                 is_input: bool = False,
                 is_output: bool = False):
        self.model = model
        self.position = position
        self.chain_id = chain_id
        self.is_input = is_input  # First model in chain
        self.is_output = is_output  # Last model in chain
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get position metadata"""
        return {
            "position": self.position,
            "chain_id": self.chain_id,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "model_config": self.model.get_composition_metadata()
        }

class CompositionChain:
    """
    Represents a chain of composable models.
    
    Handles:
    - Model sequencing and compatibility validation
    - Data flow between models
    - Chain metadata and configuration
    """
    
    def __init__(self, chain_id: str, description: str = ""):
        self.chain_id = chain_id
        self.description = description
        self.positions: List[ChainPosition] = []
        self._validated = False
    
    def add_model(self, model: ComposableModelInterface) -> 'CompositionChain':
        """
        Add a model to the end of the chain.
        
        Args:
            model: ComposableModelInterface instance
            
        Returns:
            Self for method chaining
        """
        position = len(self.positions)
        is_input = position == 0
        
        chain_position = ChainPosition(
            model=model,
            position=position,
            chain_id=self.chain_id,
            is_input=is_input,
            is_output=False  # Will be updated when chain is finalized
        )
        
        self.positions.append(chain_position)
        self._validated = False  # Need to revalidate
        
        return self
    
    def finalize(self) -> bool:
        """
        Finalize the chain by validating compatibility and marking the last model.
        
        Returns:
            True if chain is valid, False otherwise
        """
        if len(self.positions) == 0:
            return False
        
        # Mark the last model as output
        self.positions[-1].is_output = True
        
        # Validate compatibility between adjacent models
        for i in range(len(self.positions) - 1):
            current_model = self.positions[i].model
            next_model = self.positions[i + 1].model
            
            if not current_model.is_compatible_with(next_model):
                print(f"❌ Compatibility error between position {i} and {i+1}")
                print(f"   {current_model.get_config()['name']} -> {next_model.get_config()['name']}")
                print(f"   Output shape: {current_model.get_output_shape()}")
                print(f"   Expected input shape: {next_model.get_input_shape()}")
                return False
        
        self._validated = True
        return True
    
    def is_valid(self) -> bool:
        """Check if chain is valid and finalized"""
        return self._validated and len(self.positions) > 0
    
    def get_models(self) -> List[ComposableModelInterface]:
        """Get list of models in chain order"""
        return [pos.model for pos in self.positions]
    
    def get_chain_config(self) -> Dict[str, Any]:
        """Get complete chain configuration"""
        return {
            "chain_id": self.chain_id,
            "description": self.description,
            "length": len(self.positions),
            "validated": self._validated,
            "positions": [pos.get_metadata() for pos in self.positions]
        }
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, index: int) -> ChainPosition:
        return self.positions[index]
    
    def __iter__(self):
        return iter(self.positions)