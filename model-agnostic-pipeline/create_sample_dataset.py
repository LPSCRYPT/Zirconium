#!/usr/bin/env python3
"""
Create a larger sample dataset for GAN training demonstration
"""

import os
import numpy as np
from PIL import Image

def create_diverse_sample_dataset(dataset_path="gan_models/datasets/sample_training", num_images=100):
    """Create a diverse set of sample images for training"""
    os.makedirs(dataset_path, exist_ok=True)
    
    print(f"🎨 Creating {num_images} diverse sample images...")
    
    for i in range(num_images):
        # Create 32x32 images (will be resized to 16x16 during training)
        img_array = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Create different pattern types
        pattern_type = i % 6
        
        if pattern_type == 0:  # Solid colors with gradients
            base_color = [
                int(255 * (i % 8) / 8),           # Red component
                int(255 * ((i // 8) % 8) / 8),    # Green component  
                int(255 * ((i // 16) % 8) / 8)    # Blue component
            ]
            for x in range(32):
                for y in range(32):
                    # Add gradient effect
                    factor = (x + y) / 64.0
                    img_array[x, y] = [
                        min(255, int(base_color[0] * (0.5 + factor * 0.5))),
                        min(255, int(base_color[1] * (0.5 + factor * 0.5))),
                        min(255, int(base_color[2] * (0.5 + factor * 0.5)))
                    ]
        
        elif pattern_type == 1:  # Checkerboard patterns
            size = 2 + (i % 6)  # Varying checkerboard size
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green  
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255]   # Cyan
            ]
            color1 = colors[i % 6]
            color2 = colors[(i + 3) % 6]
            
            for x in range(32):
                for y in range(32):
                    if (x // size + y // size) % 2 == 0:
                        img_array[x, y] = color1
                    else:
                        img_array[x, y] = color2
        
        elif pattern_type == 2:  # Circular patterns
            center_x, center_y = 16, 16
            colors = [
                [255, 100, 100],  # Light red
                [100, 255, 100],  # Light green
                [100, 100, 255],  # Light blue
            ]
            base_color = colors[i % 3]
            
            for x in range(32):
                for y in range(32):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    intensity = max(0, 1 - distance / 16)
                    img_array[x, y] = [
                        int(base_color[0] * intensity),
                        int(base_color[1] * intensity),
                        int(base_color[2] * intensity)
                    ]
        
        elif pattern_type == 3:  # Striped patterns
            direction = i % 4  # 0=horizontal, 1=vertical, 2=diagonal1, 3=diagonal2
            stripe_width = 2 + (i % 4)
            
            for x in range(32):
                for y in range(32):
                    if direction == 0:    # Horizontal stripes
                        stripe_pos = y // stripe_width
                    elif direction == 1:  # Vertical stripes  
                        stripe_pos = x // stripe_width
                    elif direction == 2:  # Diagonal stripes
                        stripe_pos = (x + y) // stripe_width
                    else:                 # Anti-diagonal stripes
                        stripe_pos = (x - y + 32) // stripe_width
                    
                    if stripe_pos % 2 == 0:
                        img_array[x, y] = [200, 50, 200]   # Purple
                    else:
                        img_array[x, y] = [50, 200, 50]    # Green
        
        elif pattern_type == 4:  # Random noise with structure
            base_color = [
                100 + (i * 37) % 155,   # Varying base colors
                100 + (i * 73) % 155,
                100 + (i * 109) % 155
            ]
            
            for x in range(32):
                for y in range(32):
                    # Add structured noise
                    noise = np.random.randint(-50, 50)
                    img_array[x, y] = [
                        max(0, min(255, base_color[0] + noise)),
                        max(0, min(255, base_color[1] + noise)),
                        max(0, min(255, base_color[2] + noise))
                    ]
        
        else:  # Complex geometric patterns
            # Create mandala-like patterns
            center_x, center_y = 16, 16
            
            for x in range(32):
                for y in range(32):
                    dx, dy = x - center_x, y - center_y
                    angle = np.arctan2(dy, dx)
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Create radial pattern
                    radial_factor = int(8 * angle / (2 * np.pi)) % 8
                    distance_factor = int(distance) % 4
                    
                    if (radial_factor + distance_factor) % 2 == 0:
                        img_array[x, y] = [255, 150, 0]    # Orange
                    else:
                        img_array[x, y] = [0, 150, 255]    # Light blue
        
        # Save image
        img = Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(dataset_path, f'sample_{i:03d}.png'))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_images} images...")
    
    print(f"✅ Created {num_images} diverse sample images in {dataset_path}")
    print(f"📁 Dataset ready for training!")
    
    return dataset_path

if __name__ == "__main__":
    # Create sample dataset
    dataset_path = create_diverse_sample_dataset(num_images=200)
    
    print(f"\n🚀 Ready to train! Try:")
    print(f"python gan_models/train_pipeline.py --dataset {dataset_path} --epochs 10")
    
    print(f"\n📁 Sample images created:")
    files = os.listdir(dataset_path)
    print(f"  {len(files)} images: {files[:5]}...")
    
    print(f"\n💡 This dataset contains:")
    print(f"  - Solid colors with gradients")
    print(f"  - Checkerboard patterns") 
    print(f"  - Circular patterns")
    print(f"  - Striped patterns")
    print(f"  - Structured noise")
    print(f"  - Geometric patterns")
    print(f"  All will be resized to 16×16 RGB during training")