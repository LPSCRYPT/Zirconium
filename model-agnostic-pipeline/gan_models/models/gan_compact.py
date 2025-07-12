#!/usr/bin/env python3
"""
Compact Interesting GAN - Optimized for EZKL compatibility
Generates 16x16 RGB images with minimal parameters
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from ezkl_pipeline import ModelInterface

class CompactGANGenerator(nn.Module):
    """
    Compact GAN Generator optimized for EZKL compatibility
    Generates 16x16 RGB images (768 output values) - good balance of quality vs complexity
    """
    
    def __init__(self, noise_dim: int = 50, hidden_dim: int = 64):
        super(CompactGANGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # 4-layer architecture optimized for EZKL
        self.main = nn.Sequential(
            # Layer 1: Noise expansion
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(inplace=True),
            
            # Layer 2: Feature development  
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: Rich feature space
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            # Layer 4: Final output layer
            nn.Linear(hidden_dim * 4, 16 * 16 * 3),  # 16x16 RGB = 768 values
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        # Generate flat 768-dimensional output
        output = self.main(x)
        # Reshape to 16x16 RGB image (keeping batch dimension)
        return output.view(-1, 3, 16, 16)

class CompactGANDiscriminator(nn.Module):
    """
    Discriminator for training the CompactGAN
    Takes 16x16 RGB images and outputs real/fake probability
    """
    
    def __init__(self, hidden_dim: int = 64):
        super(CompactGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 768 values (16x16x3)
            nn.Linear(16 * 16 * 3, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being real
        )
    
    def forward(self, x):
        # Flatten 16x16x3 to 768
        x = x.view(-1, 16 * 16 * 3)
        return self.main(x)

class CompactGANModel(ModelInterface):
    """
    Compact GAN implementing ModelInterface for MAP compatibility
    Generates 16x16 RGB images from 50-dimensional noise
    Optimized for EZKL circuit size constraints
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        
    def get_model(self) -> torch.nn.Module:
        """Return the generator model for EZKL export"""
        if self._model is None:
            self._model = CompactGANGenerator()
            self._model.eval()
        return self._model
    
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input tensor for ONNX export"""
        return torch.randn(1, 50)  # Batch size 1, 50-dim noise
    
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL format (nested list)"""
        # Generate reproducible noise for verification
        torch.manual_seed(42)
        sample_input = torch.randn(1, 50)
        
        # EZKL requires nested list format
        return sample_input.tolist()
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        param_count = sum(p.numel() for p in self.get_model().parameters())
        return {
            "name": "gan_compact",
            "description": "4-layer compact GAN generating 16x16 RGB images",
            "architecture": "4-layer MLP Generator",
            "domain": "Generative Modeling",
            "input_shape": [1, 50],
            "output_shape": [1, 3, 16, 16],
            "parameters": param_count,
            "status": "🧪 EXPERIMENTAL"
        }
    
    def generate_sample_image(self) -> torch.Tensor:
        """Generate a sample image using the model"""
        model = self.get_model()
        
        # Use fixed seed for reproducible generation
        torch.manual_seed(42)
        noise = torch.randn(1, 50)
        
        with torch.no_grad():
            generated_image = model(noise)
        
        return generated_image
    
    def get_training_components(self):
        """Return both generator and discriminator for training"""
        generator = CompactGANGenerator()
        discriminator = CompactGANDiscriminator()
        return generator, discriminator

# Example usage and testing
if __name__ == "__main__":
    print("🎨 Testing Compact GAN Model")
    print("=" * 50)
    
    # Test model creation
    model = CompactGANModel()
    config = model.get_config()
    
    print(f"Model: {config['name']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Output: {config['output_shape']} (16x16 RGB)")
    print(f"Parameters: {config['parameters']:,}")
    
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
    
    print("\n✅ Compact GAN model ready for EZKL export!")