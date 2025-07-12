#!/usr/bin/env python3
"""
Visualize Compact GAN output and save as image file
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.gan_compact import CompactGANModel

def tensor_to_image(tensor, filename="gan_output.png"):
    """Convert tensor to PIL Image and save"""
    # tensor shape: [1, 3, 16, 16]
    # Remove batch dimension: [3, 16, 16]
    image_tensor = tensor.squeeze(0)
    
    # Convert from [-1, 1] to [0, 1]
    image_tensor = (image_tensor + 1) / 2
    
    # Clamp to valid range
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to numpy: [3, 16, 16] -> [16, 16, 3]
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()
    
    # Convert to 0-255 range
    image_np = (image_np * 255).astype(np.uint8)
    
    # Create PIL Image
    pil_image = Image.fromarray(image_np)
    
    # Save original 16x16
    pil_image.save(filename)
    
    # Also save upscaled version for better visibility
    upscaled = pil_image.resize((256, 256), Image.NEAREST)  # 16x upscale
    upscaled_filename = filename.replace('.png', '_upscaled.png')
    upscaled.save(upscaled_filename)
    
    return pil_image, upscaled

def create_comparison_grid():
    """Create a grid of multiple GAN outputs"""
    model = CompactGANModel()
    
    # Generate multiple samples
    samples = []
    for i in range(16):  # 4x4 grid
        torch.manual_seed(i + 42)  # Different seeds
        output = model.generate_sample_image()
        samples.append(output)
    
    # Create grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('Compact GAN Outputs (Untrained) - 16x16 RGB', fontsize=14)
    
    for i, sample in enumerate(samples):
        row, col = i // 4, i % 4
        
        # Convert tensor to displayable format
        image_tensor = sample.squeeze(0)  # Remove batch dim
        image_tensor = (image_tensor + 1) / 2  # [-1,1] to [0,1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert to numpy for matplotlib
        image_np = image_tensor.permute(1, 2, 0).detach().numpy()
        
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'Seed {i+42}', fontsize=8)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('gan_compact_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return samples

def main():
    print("🎨 Generating Compact GAN Output")
    print("=" * 50)
    
    # Create model
    model = CompactGANModel()
    config = model.get_config()
    
    print(f"Model: {config['name']}")
    print(f"Output: {config['output_shape']} (16x16 RGB)")
    print(f"Parameters: {config['parameters']:,}")
    print(f"Status: {config['status']}")
    
    # Generate single sample
    print("\n🎯 Generating single sample...")
    torch.manual_seed(42)  # Reproducible
    generated_image = model.generate_sample_image()
    
    print(f"Generated shape: {generated_image.shape}")
    print(f"Value range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
    
    # Save as image files
    print("\n💾 Saving as image files...")
    original, upscaled = tensor_to_image(generated_image, "gan_compact_single.png")
    
    print(f"✅ Saved: gan_compact_single.png (16x16 original)")
    print(f"✅ Saved: gan_compact_single_upscaled.png (256x256 upscaled)")
    
    # Create comparison grid
    print("\n🎯 Creating comparison grid...")
    samples = create_comparison_grid()
    print(f"✅ Saved: gan_compact_grid.png (4x4 grid of samples)")
    
    # Show statistics
    print("\n📊 Image Statistics:")
    flat_image = generated_image.squeeze().flatten().detach().numpy()
    print(f"   Mean: {flat_image.mean():.4f}")
    print(f"   Std:  {flat_image.std():.4f}") 
    print(f"   Min:  {flat_image.min():.4f}")
    print(f"   Max:  {flat_image.max():.4f}")
    
    # RGB channel analysis
    r_channel = generated_image[0, 0, :, :].flatten().detach().numpy()
    g_channel = generated_image[0, 1, :, :].flatten().detach().numpy()
    b_channel = generated_image[0, 2, :, :].flatten().detach().numpy()
    
    print(f"\n🎨 RGB Channel Analysis:")
    print(f"   Red   - Mean: {r_channel.mean():.3f}, Std: {r_channel.std():.3f}")
    print(f"   Green - Mean: {g_channel.mean():.3f}, Std: {g_channel.std():.3f}")
    print(f"   Blue  - Mean: {b_channel.mean():.3f}, Std: {b_channel.std():.3f}")
    
    print(f"\n🎉 Compact GAN visualization complete!")
    print(f"📁 Files created:")
    print(f"   - gan_compact_single.png (16x16)")
    print(f"   - gan_compact_single_upscaled.png (256x256)")  
    print(f"   - gan_compact_grid.png (comparison grid)")

if __name__ == "__main__":
    main()