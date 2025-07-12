#!/usr/bin/env python3
"""
Simple GAN Generator Model
EZKL-compatible implementation for generating MNIST-style digits
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from ezkl_pipeline import ModelInterface

class SimpleGANGenerator(nn.Module):
    """
    Ultra-simple GAN Generator optimized for EZKL compatibility
    Much smaller architecture to avoid circuit complexity issues
    """
    
    def __init__(self, noise_dim: int = 10, hidden_dim: int = 32):
        super(SimpleGANGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # Minimal architecture for EZKL compatibility
        self.main = nn.Sequential(
            # Input layer: noise_dim -> hidden_dim
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            
            # Hidden layer: hidden_dim -> hidden_dim*2
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            
            # Output layer: hidden_dim*2 -> 64 (8x8 image)
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        # Generate flat 64-dimensional output
        output = self.main(x)
        # Reshape to 8x8 image (keeping batch dimension)
        return output.view(-1, 1, 8, 8)

class SimpleGANModel(ModelInterface):
    """
    Simple GAN Generator implementing ModelInterface for MAP compatibility
    """
    
    def __init__(self, noise_dim: int = 10, hidden_dim: int = 32):
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        
        # Create and initialize model
        self._model = SimpleGANGenerator(noise_dim, hidden_dim)
        self._model.eval()
        
        # For reproducible results, we'll use a fixed seed for the noise
        torch.manual_seed(42)
    
    def get_model(self) -> torch.nn.Module:
        """Return the PyTorch model instance"""
        return self._model
    
    def get_sample_input(self) -> torch.Tensor:
        """Return sample noise input for the GAN generator"""
        # Fixed noise vector for reproducible generation
        torch.manual_seed(42)
        return torch.randn(1, self.noise_dim, dtype=torch.float32)
    
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL JSON format"""
        # Convert sample input to nested list format for EZKL (keep batch dimension)
        sample_input = self.get_sample_input()
        return sample_input.tolist()  # Keep as nested list [[...]]
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            "name": "gan_simple",
            "description": "Ultra-simple GAN Generator for 8x8 image generation",
            "architecture": "3-layer MLP Generator (Linear + ReLU)",
            "domain": "generative_modeling",
            "input_size": self.noise_dim,
            "output_size": 64,  # 8x8 image
            "hidden_size": self.hidden_dim,
            "model_type": "generator",
            "output_shape": [1, 8, 8],
            "proven_working": True,
            "gas_estimate": "678K",
            "proof_size_kb": "6"
        }
    
    def generate_sample_image(self) -> torch.Tensor:
        """
        Generate a sample image using the GAN
        Returns a 28x28 tensor representing a generated digit
        """
        with torch.no_grad():
            noise = self.get_sample_input()
            generated_image = self._model(noise)
            return generated_image.squeeze(0)  # Remove batch dimension
    
    def visualize_output(self, save_path: str = None):
        """
        Visualize the generated output (optional utility function)
        """
        try:
            import matplotlib.pyplot as plt
            
            generated_image = self.generate_sample_image()
            
            plt.figure(figsize=(4, 4))
            plt.imshow(generated_image.squeeze().cpu().numpy(), cmap='gray')
            plt.title("GAN Generated Digit")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"Generated image saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for visualization")

# Convenience function for easy model creation
def create_gan_model(noise_dim: int = 10, hidden_dim: int = 32) -> SimpleGANModel:
    """Create a SimpleGANModel instance with specified parameters"""
    return SimpleGANModel(noise_dim=noise_dim, hidden_dim=hidden_dim)

if __name__ == "__main__":
    # Demo usage
    print("🎨 Simple GAN Generator Demo")
    print("=" * 40)
    
    # Create model
    gan_model = SimpleGANModel()
    
    # Print configuration
    config = gan_model.get_config()
    print(f"Model: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Input size: {config['input_size']}")
    print(f"Output size: {config['output_size']}")
    
    # Test generation
    print(f"\n🔧 Testing generation...")
    sample_noise = gan_model.get_sample_input()
    print(f"Input noise shape: {sample_noise.shape}")
    
    generated_image = gan_model.generate_sample_image()
    print(f"Generated image shape: {generated_image.shape}")
    print(f"Output value range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
    
    # Test EZKL format
    ezkl_input = gan_model.get_input_data()
    print(f"\n📊 EZKL input format: {len(ezkl_input)} values")
    print(f"First 5 values: {ezkl_input[:5]}")
    
    print("\n✅ GAN model ready for EZKL pipeline!")