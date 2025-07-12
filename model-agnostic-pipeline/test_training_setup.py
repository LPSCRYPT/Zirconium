#!/usr/bin/env python3
"""
Test GAN training setup without actually training
Verifies the pipeline works with sample data
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add paths
sys.path.append('gan_models')
sys.path.append('gan_models/models')

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    dataset_path = "gan_models/datasets/test_images"
    os.makedirs(dataset_path, exist_ok=True)
    
    print("🎨 Creating sample test images...")
    
    # Create 10 sample 16x16 RGB images
    for i in range(10):
        # Create random colored pattern
        img_array = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        
        # Add some structure (not pure noise)
        if i % 3 == 0:  # Some checkerboard patterns
            for x in range(16):
                for y in range(16):
                    if (x + y) % 4 < 2:
                        img_array[x, y] = [255, 0, 0]  # Red squares
        elif i % 3 == 1:  # Some gradient patterns  
            for x in range(16):
                for y in range(16):
                    img_array[x, y] = [x*16, y*16, 128]
        # else: keep random
        
        # Save as PNG
        img = Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(dataset_path, f'sample_{i:03d}.png'))
    
    print(f"✅ Created {len(os.listdir(dataset_path))} sample images")
    return dataset_path

def test_dataset_loading():
    """Test that the training pipeline can load the dataset"""
    try:
        from train_pipeline import CustomImageDataset, GANTrainingPipeline, create_training_config
        import torchvision.transforms as transforms
        
        print("📦 Testing dataset loading...")
        
        # Create sample dataset
        dataset_path = create_sample_dataset()
        
        # Test dataset class
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = CustomImageDataset(dataset_path, transform=transform)
        print(f"✅ Dataset loaded: {len(dataset)} images")
        
        # Test loading a sample
        sample_image, _ = dataset[0]
        print(f"✅ Sample image shape: {sample_image.shape}")
        print(f"✅ Sample image range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
        
        # Test training pipeline setup (without training)
        config = create_training_config("test_run")
        config['num_epochs'] = 1  # Just 1 epoch for testing
        
        pipeline = GANTrainingPipeline(config)
        print("✅ Training pipeline initialized")
        
        # Test dataloader setup
        dataloader = pipeline.setup_dataset(dataset_path)
        print(f"✅ Dataloader created: {len(dataloader)} batches")
        
        # Test one batch
        for batch_images, _ in dataloader:
            print(f"✅ Batch loaded: {batch_images.shape}")
            break
            
        print("\n🎉 Training setup test PASSED!")
        print(f"📁 Sample dataset created: {dataset_path}")
        print("\n🚀 Ready to train! Use:")
        print(f"   python gan_models/train_pipeline.py --dataset {dataset_path} --epochs 5")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_training_commands():
    """Show example training commands"""
    print("\n" + "="*60)
    print("🎯 GAN Training Commands")
    print("="*60)
    
    print("\n📝 Quick Test Training (5 epochs):")
    print("python gan_models/train_pipeline.py --dataset gan_models/datasets/test_images --epochs 5")
    
    print("\n📝 Full Training with Your Images:")
    print("# 1. Copy your images:")
    print("cp /path/to/your/images/* gan_models/datasets/my_images/")
    print("# 2. Start training:")
    print("python gan_models/train_pipeline.py --dataset gan_models/datasets/my_images --epochs 50")
    
    print("\n📝 Custom Training:")
    print("python gan_models/train_pipeline.py \\")
    print("  --dataset gan_models/datasets/my_images \\")
    print("  --experiment my_custom_gan \\")
    print("  --epochs 100 \\")
    print("  --batch-size 64 \\")
    print("  --lr 0.0001")
    
    print("\n📝 Resume Training:")
    print("python gan_models/train_pipeline.py \\")
    print("  --dataset gan_models/datasets/my_images \\")
    print("  --resume gan_models/checkpoints/my_custom_gan/latest.pth")
    
    print("\n📝 Monitor Training:")
    print("# Watch generated samples:")
    print("ls gan_models/outputs/EXPERIMENT_NAME/samples/")
    print("# Check training logs:")
    print("tail -f gan_models/logs/EXPERIMENT_NAME/training_log.jsonl")

if __name__ == "__main__":
    print("🧪 Testing GAN Training Pipeline Setup")
    print("="*50)
    
    success = test_dataset_loading()
    show_training_commands()
    
    if success:
        print("\n✅ All tests passed! Training pipeline is ready.")
    else:
        print("\n❌ Setup test failed. Check the error messages above.")