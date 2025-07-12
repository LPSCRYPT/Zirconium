#!/usr/bin/env python3
"""
Interesting GAN Model - 5-layer architecture for 32x32 RGB generation
EZKL-compatible but capable of generating recognizable images
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from ezkl_pipeline import ModelInterface

class InterestingGANGenerator(nn.Module):
    """
    5-layer GAN Generator optimized for balance between EZKL compatibility and visual quality
    Generates 32x32 RGB images (3,072 output values)
    """
    
    def __init__(self, noise_dim: int = 100, hidden_dim: int = 256):
        super(InterestingGANGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # 5-layer architecture for interesting outputs
        self.main = nn.Sequential(
            # Layer 1: Noise expansion
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Layer 2: Feature development  
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: Rich feature space
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            # Layer 4: Pre-output expansion
            nn.Linear(hidden_dim * 4, hidden_dim * 6),
            nn.BatchNorm1d(hidden_dim * 6),
            nn.ReLU(inplace=True),
            
            # Layer 5: Final output layer
            nn.Linear(hidden_dim * 6, 32 * 32 * 3),  # 32x32 RGB = 3,072 values
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        # Generate flat 3,072-dimensional output
        output = self.main(x)
        # Reshape to 32x32 RGB image (keeping batch dimension)
        return output.view(-1, 3, 32, 32)

class InterestingGANDiscriminator(nn.Module):
    """
    Discriminator for training the InterestingGAN
    Takes 32x32 RGB images and outputs real/fake probability
    """
    
    def __init__(self, hidden_dim: int = 256):
        super(InterestingGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 3,072 values (32x32x3)
            nn.Linear(32 * 32 * 3, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being real
        )
    
    def forward(self, x):
        # Flatten 32x32x3 to 3,072
        x = x.view(-1, 32 * 32 * 3)
        return self.main(x)

class InterestingGANModel(ModelInterface):
    """
    Interesting GAN implementing ModelInterface for MAP compatibility
    Generates 32x32 RGB images from 100-dimensional noise
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        
    def get_model(self) -> torch.nn.Module:
        """Return the generator model for EZKL export"""
        if self._model is None:
            self._model = InterestingGANGenerator()
            self._model.eval()
        return self._model
    
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input tensor for ONNX export"""
        return torch.randn(1, 100)  # Batch size 1, 100-dim noise
    
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL format (nested list)"""
        # Generate reproducible noise for verification
        torch.manual_seed(42)
        sample_input = torch.randn(1, 100)
        
        # EZKL requires nested list format
        return sample_input.tolist()
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            "name": "gan_interesting",
            "description": "5-layer GAN generating 32x32 RGB images",
            "architecture": "5-layer MLP Generator",
            "domain": "Generative Modeling",
            "input_shape": [1, 100],
            "output_shape": [1, 3, 32, 32],
            "parameters": 531_200,  # Approximate parameter count
            "status": "🧪 EXPERIMENTAL"
        }
    
    def generate_sample_image(self) -> torch.Tensor:
        """Generate a sample image using the model"""
        model = self.get_model()
        
        # Use fixed seed for reproducible generation
        torch.manual_seed(42)
        noise = torch.randn(1, 100)
        
        with torch.no_grad():
            generated_image = model(noise)
        
        return generated_image
    
    def get_training_components(self):
        """Return both generator and discriminator for training"""
        generator = InterestingGANGenerator()
        discriminator = InterestingGANDiscriminator()
        return generator, discriminator

# Example usage and testing
if __name__ == "__main__":
    print("🎨 Testing Interesting GAN Model")
    print("=" * 50)
    
    # Test model creation
    model = InterestingGANModel()
    config = model.get_config()
    
    print(f"Model: {config['name']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Output: {config['output_shape']} (32x32 RGB)")
    print(f"Parameters: ~{config['parameters']:,}")
    
    # Test forward pass
    generator = model.get_model()
    sample_input = model.get_sample_input()
    
    print(f"\nInput shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = generator(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test EZKL format
    ezkl_input = model.get_input_data()
    print(f"EZKL input format: {len(ezkl_input)}x{len(ezkl_input[0])} values")
    
    print("\n✅ Interesting GAN model ready for training and EZKL export!")