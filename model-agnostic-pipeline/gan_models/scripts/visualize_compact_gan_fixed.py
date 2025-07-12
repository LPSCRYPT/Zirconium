#!/usr/bin/env python3
"""
Fixed Compact GAN visualization with properly different seeds
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.gan_compact import CompactGANModel

def generate_with_seed(model, seed):
    """Generate image with specific seed"""
    torch.manual_seed(seed)
    noise = torch.randn(1, 50)
    
    with torch.no_grad():
        generated_image = model.get_model()(noise)
    
    return generated_image

def create_comparison_grid_fixed():
    """Create a grid of multiple GAN outputs with actually different seeds"""
    model = CompactGANModel()
    
    # Generate multiple samples with different seeds
    samples = []
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        output = generate_with_seed(model, seed)
        samples.append(output)
    
    # Create grid
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Compact GAN Outputs (Untrained) - Different Seeds - 16x16 RGB', fontsize=14)
    
    for i, (sample, seed) in enumerate(zip(samples, seeds)):
        row, col = i // 4, i % 4
        
        # Convert tensor to displayable format
        image_tensor = sample.squeeze(0)  # Remove batch dim
        image_tensor = (image_tensor + 1) / 2  # [-1,1] to [0,1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert to numpy for matplotlib
        image_np = image_tensor.permute(1, 2, 0).detach().numpy()
        
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'Seed {seed}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('gan_compact_grid_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return samples, seeds

def verify_different_outputs():
    """Verify that different seeds actually produce different outputs"""
    model = CompactGANModel()
    
    print("🔍 Verifying different seeds produce different outputs...")
    
    # Generate with 3 different seeds
    seeds = [42, 123, 456]
    outputs = []
    
    for seed in seeds:
        output = generate_with_seed(model, seed)
        outputs.append(output)
        
        # Show first few pixel values
        flat = output.flatten()[:10]
        print(f"Seed {seed}: First 10 values = {flat.detach().numpy()}")
    
    # Check if outputs are actually different
    diff_12 = torch.sum((outputs[0] - outputs[1])**2).item()
    diff_13 = torch.sum((outputs[0] - outputs[2])**2).item()
    diff_23 = torch.sum((outputs[1] - outputs[2])**2).item()
    
    print(f"\n📊 L2 Differences:")
    print(f"   Seed 42 vs 123: {diff_12:.6f}")
    print(f"   Seed 42 vs 456: {diff_13:.6f}")
    print(f"   Seed 123 vs 456: {diff_23:.6f}")
    
    if diff_12 > 0.001 and diff_13 > 0.001 and diff_23 > 0.001:
        print("✅ Different seeds produce clearly different outputs!")
        return True
    else:
        print("❌ Outputs are too similar - there might be a bug")
        return False

def main():
    print("🎨 Fixed Compact GAN Visualization")
    print("=" * 50)
    
    # First verify seeds work
    if verify_different_outputs():
        print("\n🎯 Creating properly different comparison grid...")
        samples, seeds = create_comparison_grid_fixed()
        print(f"✅ Saved: gan_compact_grid_fixed.png")
        print(f"📊 Generated {len(samples)} different images with seeds: {seeds[:5]}...")
    else:
        print("❌ Seed verification failed")

if __name__ == "__main__":
    main()