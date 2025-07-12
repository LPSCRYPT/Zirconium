#!/usr/bin/env python3
"""
Download and setup pre-trained GAN models
Provides options for getting trained weights for the compact GAN
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import requests
from models.gan_compact import CompactGANGenerator, CompactGANDiscriminator

def create_simple_trained_weights():
    """
    Create a minimally trained GAN for demonstration
    This trains on a small subset of CIFAR-10 for just a few epochs
    """
    print("🎯 Creating minimally trained GAN weights...")
    
    # Setup data (small subset for quick training)
    transform = transforms.Compose([
        transforms.Resize(16),  # Resize to 16x16
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Use only first 1000 samples for quick training
    subset_indices = list(range(1000))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    # Create models
    generator = CompactGANGenerator()
    discriminator = CompactGANDiscriminator()
    
    # Simple training loop (just 5 epochs for demo)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = torch.nn.BCELoss()
    
    print("🚀 Quick training (5 epochs on 1000 samples)...")
    
    for epoch in range(5):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, 50)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}: G_Loss = {g_loss.item():.4f}, D_Loss = {d_loss.item():.4f}")
    
    # Save the trained weights
    os.makedirs('pretrained_weights', exist_ok=True)
    torch.save(generator.state_dict(), 'pretrained_weights/compact_gan_generator_trained.pth')
    torch.save(discriminator.state_dict(), 'pretrained_weights/compact_gan_discriminator_trained.pth')
    
    print("✅ Minimal training completed!")
    print("📁 Weights saved: pretrained_weights/compact_gan_generator_trained.pth")
    
    return generator

def download_dcgan_weights():
    """
    Download pre-trained DCGAN weights (if available)
    This would need to be adapted from existing DCGAN models
    """
    print("💡 Pre-trained DCGAN download options:")
    print("   1. PyTorch Hub: torch.hub.load('pytorch/vision', 'dcgan', pretrained=True)")
    print("   2. Hugging Face: huggingface.co/models (search for DCGAN)")
    print("   3. Papers with Code: paperswithcode.com/task/image-generation")
    print("   4. GitHub repositories with pre-trained weights")
    
    print("\n🔄 For this demo, we'll create minimal training instead.")
    return None

def load_trained_gan():
    """
    Load a trained GAN model if weights exist
    """
    weights_path = 'pretrained_weights/compact_gan_generator_trained.pth'
    
    if os.path.exists(weights_path):
        print(f"✅ Loading trained weights from: {weights_path}")
        generator = CompactGANGenerator()
        generator.load_state_dict(torch.load(weights_path))
        generator.eval()
        return generator
    else:
        print("❌ No trained weights found. Options:")
        print("   1. Run: python download_pretrained_gan.py --train")
        print("   2. Train manually: python train_interesting_gan.py") 
        return None

def test_trained_gan(generator):
    """
    Test the trained GAN and show sample outputs
    """
    print("🎨 Testing trained GAN...")
    
    with torch.no_grad():
        # Generate a few samples
        for i in range(3):
            noise = torch.randn(1, 50)
            generated = generator(noise)
            
            # Convert from [-1, 1] to [0, 1]
            generated = (generated + 1) / 2
            
            print(f"Sample {i+1}: shape {generated.shape}, range [{generated.min():.3f}, {generated.max():.3f}]")
    
    print("✅ Trained GAN ready for EZKL verification!")

def integrate_with_ezkl():
    """
    Create a model file that uses trained weights with EZKL
    """
    print("🔧 Creating EZKL-compatible trained model...")
    
    trained_model_code = '''
from models.gan_compact import CompactGANModel, CompactGANGenerator
import torch
import os

class TrainedCompactGANModel(CompactGANModel):
    """
    Compact GAN with pre-trained weights for EZKL verification
    """
    
    def get_model(self):
        if self._model is None:
            self._model = CompactGANGenerator()
            
            # Load trained weights if available
            weights_path = 'pretrained_weights/compact_gan_generator_trained.pth'
            if os.path.exists(weights_path):
                self._model.load_state_dict(torch.load(weights_path))
                print("✅ Loaded trained weights!")
            else:
                print("⚠️ Using random weights (no training found)")
            
            self._model.eval()
        return self._model
    
    def get_config(self):
        config = super().get_config()
        config["name"] = "gan_compact_trained"
        config["description"] = "Trained compact GAN generating 16x16 RGB images"
        config["status"] = "🎯 TRAINED"
        return config
'''
    
    with open('models/gan_compact_trained.py', 'w') as f:
        f.write(trained_model_code)
    
    print("📁 Created: models/gan_compact_trained.py")
    print("💡 Register this in model_registry.py to use with demo.py")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download/create pre-trained GAN')
    parser.add_argument('--train', action='store_true', help='Create minimally trained weights')
    parser.add_argument('--load', action='store_true', help='Load existing trained weights')
    parser.add_argument('--test', action='store_true', help='Test trained model')
    parser.add_argument('--integrate', action='store_true', help='Create EZKL integration')
    
    args = parser.parse_args()
    
    print("🎨 GAN Pre-trained Model Setup")
    print("=" * 50)
    
    if args.train:
        generator = create_simple_trained_weights()
        test_trained_gan(generator)
        
    elif args.load:
        generator = load_trained_gan()
        if generator:
            test_trained_gan(generator)
            
    elif args.test:
        generator = load_trained_gan()
        if generator:
            test_trained_gan(generator)
            
    elif args.integrate:
        integrate_with_ezkl()
        
    else:
        print("🔍 Available options:")
        print("   --train     : Quick training on CIFAR-10 subset")
        print("   --load      : Load existing trained weights")  
        print("   --test      : Test trained model")
        print("   --integrate : Create EZKL integration")
        print()
        print("💡 Recommendation: Start with --train for demo purposes")

if __name__ == "__main__":
    main()