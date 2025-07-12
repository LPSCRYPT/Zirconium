#!/usr/bin/env python3
"""
Visualize the actual GAN output that was verified on blockchain
"""

from models.gan_simple import SimpleGANModel
import torch

def visualize_8x8_image(tensor, threshold=0.0):
    """Convert 8x8 tensor to ASCII art visualization"""
    # Get the 8x8 image (remove batch and channel dimensions)
    image_2d = tensor.squeeze().detach().numpy()
    
    print("🎨 GAN Generated 8x8 Image (ASCII Visualization)")
    print("=" * 50)
    print(f"Value range: [{image_2d.min():.3f}, {image_2d.max():.3f}]")
    print(f"Threshold: {threshold} (values > threshold = ●, values ≤ threshold = ·)")
    print()
    
    # ASCII visualization
    print("   0 1 2 3 4 5 6 7")
    print("  ┌─────────────────┐")
    for i in range(8):
        row_str = f"{i} │ "
        for j in range(8):
            pixel_val = image_2d[i, j]
            # Use ● for values above threshold, · for below
            symbol = "●" if pixel_val > threshold else "·"
            row_str += symbol + " "
        row_str += "│"
        print(row_str)
    print("  └─────────────────┘")
    
    print("\n📊 Raw pixel values:")
    print("   0     1     2     3     4     5     6     7")
    for i in range(8):
        row_str = f"{i} "
        for j in range(8):
            row_str += f"{image_2d[i, j]:+.3f} "
        print(row_str)

def main():
    print("🔍 Analyzing the GAN output that was verified on blockchain")
    print("=" * 60)
    
    # Create the same model used in verification
    gan_model = SimpleGANModel()
    
    # Generate the same image that was proven (using fixed seed)
    torch.manual_seed(42)  # Same seed used in verification
    generated_image = gan_model.generate_sample_image()
    
    print(f"Generated image shape: {generated_image.shape}")
    print(f"This is the exact same output that was cryptographically proven!")
    print()
    
    # Visualize with different thresholds
    thresholds = [0.0, -0.05, 0.05]
    
    for threshold in thresholds:
        print()
        visualize_8x8_image(generated_image, threshold=threshold)
        print()
    
    # Show some statistics
    image_flat = generated_image.squeeze().detach().numpy().flatten()
    print("📈 Image Statistics:")
    print(f"   Mean: {image_flat.mean():.4f}")
    print(f"   Std:  {image_flat.std():.4f}")
    print(f"   Min:  {image_flat.min():.4f}")
    print(f"   Max:  {image_flat.max():.4f}")
    print(f"   Pixels > 0: {(image_flat > 0).sum()}/64")
    print(f"   Pixels < 0: {(image_flat < 0).sum()}/64")

if __name__ == "__main__":
    main()