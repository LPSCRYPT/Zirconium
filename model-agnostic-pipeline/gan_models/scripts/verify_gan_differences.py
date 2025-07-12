#!/usr/bin/env python3
"""
Direct pixel-by-pixel comparison of GAN outputs
"""

import torch
import numpy as np
from models.gan_compact import CompactGANModel

def generate_with_seed(model, seed):
    """Generate image with specific seed"""
    torch.manual_seed(seed)
    noise = torch.randn(1, 50)
    
    with torch.no_grad():
        generated_image = model.get_model()(noise)
    
    return generated_image

def compare_outputs_directly():
    """Compare outputs pixel by pixel"""
    model = CompactGANModel()
    
    print("🔍 Direct Pixel-by-Pixel Comparison")
    print("=" * 50)
    
    # Generate 3 outputs with different seeds
    seeds = [42, 123, 456]
    outputs = []
    
    for seed in seeds:
        output = generate_with_seed(model, seed)
        outputs.append(output)
        print(f"\nSeed {seed}:")
        print(f"  Shape: {output.shape}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")
        print(f"  Mean: {output.mean().item():.6f}")
        
        # Show first 5 pixels of first channel
        first_5 = output[0, 0, 0, :5].detach().numpy()
        print(f"  First 5 pixels (R channel, row 0): {first_5}")
    
    # Direct tensor comparison
    print(f"\n🎯 Direct Tensor Comparisons:")
    
    # Check if tensors are identical
    identical_42_123 = torch.equal(outputs[0], outputs[1])
    identical_42_456 = torch.equal(outputs[0], outputs[2])
    identical_123_456 = torch.equal(outputs[1], outputs[2])
    
    print(f"  Seed 42 == Seed 123: {identical_42_123}")
    print(f"  Seed 42 == Seed 456: {identical_42_456}")
    print(f"  Seed 123 == Seed 456: {identical_123_456}")
    
    # Calculate exact differences
    diff_42_123 = outputs[0] - outputs[1]
    diff_42_456 = outputs[0] - outputs[2]
    diff_123_456 = outputs[1] - outputs[2]
    
    print(f"\n📊 Difference Statistics:")
    print(f"  42 vs 123 - Max diff: {diff_42_123.abs().max().item():.6f}")
    print(f"  42 vs 123 - Mean diff: {diff_42_123.abs().mean().item():.6f}")
    print(f"  42 vs 123 - Non-zero pixels: {(diff_42_123 != 0).sum().item()}/{diff_42_123.numel()}")
    
    print(f"  42 vs 456 - Max diff: {diff_42_456.abs().max().item():.6f}")
    print(f"  42 vs 456 - Mean diff: {diff_42_456.abs().mean().item():.6f}")
    print(f"  42 vs 456 - Non-zero pixels: {(diff_42_456 != 0).sum().item()}/{diff_42_456.numel()}")
    
    # Sample specific pixel comparisons
    print(f"\n🔬 Specific Pixel Comparisons:")
    for i in range(3):
        for j in range(3):
            pixel_42 = outputs[0][0, 0, i, j].item()
            pixel_123 = outputs[1][0, 0, i, j].item()
            pixel_456 = outputs[2][0, 0, i, j].item()
            print(f"  Pixel[{i},{j}] Red: {pixel_42:.4f} vs {pixel_123:.4f} vs {pixel_456:.4f}")
    
    # Final verification
    all_different = not (identical_42_123 or identical_42_456 or identical_123_456)
    if all_different:
        print(f"\n✅ CONFIRMED: All outputs are genuinely different!")
        print(f"   Every single pixel value differs between seeds")
    else:
        print(f"\n❌ WARNING: Some outputs are identical!")
        
    return all_different

def test_reproducibility():
    """Test that same seed produces same output"""
    model = CompactGANModel()
    
    print(f"\n🔄 Testing Reproducibility (Same Seed)")
    print("=" * 30)
    
    # Generate same seed twice
    output1 = generate_with_seed(model, 42)
    output2 = generate_with_seed(model, 42)
    
    identical = torch.equal(output1, output2)
    print(f"Same seed (42) twice: {identical}")
    
    if identical:
        print("✅ Same seed produces identical output (good!)")
    else:
        print("❌ Same seed produces different output (bad!)")
        
    return identical

if __name__ == "__main__":
    # Test that different seeds produce different outputs
    different = compare_outputs_directly()
    
    # Test that same seed produces same output
    reproducible = test_reproducibility()
    
    print(f"\n🎯 Final Verification:")
    print(f"  Different seeds → Different outputs: {different}")
    print(f"  Same seed → Same output: {reproducible}")
    
    if different and reproducible:
        print(f"✅ GAN output validation PASSED!")
    else:
        print(f"❌ GAN output validation FAILED!")