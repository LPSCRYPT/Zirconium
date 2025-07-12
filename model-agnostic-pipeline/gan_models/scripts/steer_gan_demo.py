#!/usr/bin/env python3
"""
Demonstration of steering the 8x8 GAN towards specific output patterns
"""

import torch
from models.gan_simple import SimpleGANModel
import numpy as np

def visualize_pattern(tensor, title="Pattern"):
    """Convert tensor to ASCII visualization"""
    image_2d = tensor.squeeze().detach().numpy()
    print(f"\n{title}:")
    print("   0 1 2 3 4 5 6 7")
    print("  ┌─────────────────┐")
    for i in range(8):
        row_str = f"{i} │ "
        for j in range(8):
            symbol = "●" if image_2d[i, j] > 0 else "·"
            row_str += symbol + " "
        row_str += "│"
        print(row_str)
    print("  └─────────────────┘")

def create_target_patterns():
    """Create some simple target patterns"""
    
    # Pattern 1: Checkerboard
    checkerboard = torch.zeros(1, 1, 8, 8)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                checkerboard[0, 0, i, j] = 0.5
            else:
                checkerboard[0, 0, i, j] = -0.5
    
    # Pattern 2: Cross
    cross = torch.full((1, 1, 8, 8), -0.3)
    cross[0, 0, 3:5, :] = 0.5  # Horizontal line
    cross[0, 0, :, 3:5] = 0.5  # Vertical line
    
    # Pattern 3: Border
    border = torch.full((1, 1, 8, 8), -0.3)
    border[0, 0, 0, :] = 0.5   # Top
    border[0, 0, 7, :] = 0.5   # Bottom  
    border[0, 0, :, 0] = 0.5   # Left
    border[0, 0, :, 7] = 0.5   # Right
    
    return {
        "checkerboard": checkerboard,
        "cross": cross,  
        "border": border
    }

def random_search_steering(gan, target, iterations=500):
    """Search random inputs to find closest match to target"""
    print(f"\n🔍 Random search for {iterations} iterations...")
    
    best_input = None
    best_loss = float('inf')
    best_output = None
    
    for i in range(iterations):
        # Try random input
        noise = torch.randn(1, 10)
        output = gan._model(noise)
        
        # Calculate similarity to target
        loss = torch.nn.functional.mse_loss(output, target).item()
        
        if loss < best_loss:
            best_loss = loss
            best_input = noise.clone()
            best_output = output.clone()
            
        if i % 100 == 0:
            print(f"  Iteration {i}: Best loss so far = {best_loss:.4f}")
    
    return best_input, best_output, best_loss

def main():
    print("🎛️ Steering the 8x8 GAN Demonstration")
    print("=" * 50)
    
    # Create GAN model
    gan = SimpleGANModel()
    
    # Show original verified output
    torch.manual_seed(42)
    original_output = gan.generate_sample_image()
    visualize_pattern(original_output, "🎨 Original Verified Output (seed=42)")
    
    # Create target patterns
    targets = create_target_patterns()
    
    # Test steering towards each target
    for pattern_name, target in targets.items():
        print(f"\n{'='*50}")
        print(f"🎯 Attempting to steer towards: {pattern_name.upper()}")
        
        # Show target
        visualize_pattern(target, f"Target: {pattern_name}")
        
        # Search for best input
        best_input, best_output, best_loss = random_search_steering(gan, target)
        
        # Show result
        visualize_pattern(best_output, f"Best Match (loss={best_loss:.4f})")
        
        print(f"📊 Best input vector: {best_input.squeeze()[:5].tolist()}... (first 5 values)")
        print(f"📈 Final loss: {best_loss:.4f}")
        
        # Compare with random output
        random_noise = torch.randn(1, 10)
        random_output = gan._model(random_noise)
        random_loss = torch.nn.functional.mse_loss(random_output, target).item()
        
        improvement = random_loss / best_loss
        print(f"🚀 Improvement over random: {improvement:.2f}x better")
    
    print("\n" + "="*50)
    print("📝 Summary:")
    print("✅ CAN steer GAN towards specific patterns")
    print("❌ Limited by untrained nature (random patterns only)")
    print("❌ 8x8 resolution severely limits complexity")
    print("💡 Shows proof-of-concept for input space exploration")

if __name__ == "__main__":
    main()