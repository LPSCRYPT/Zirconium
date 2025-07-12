#!/usr/bin/env python3
"""
Training script for the Interesting GAN
Trains on CIFAR-10 dataset to generate 32x32 RGB images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from models.gan_interesting import InterestingGANGenerator, InterestingGANDiscriminator

def visualize_samples(generator, epoch, device, num_samples=16):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(num_samples, 100, device=device)
        fake_images = generator(noise)
        
        # Convert from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Generated Samples - Epoch {epoch}')
        
        for i in range(num_samples):
            row, col = i // 4, i % 4
            # Convert from CHW to HWC for matplotlib
            img = fake_images[i].cpu().permute(1, 2, 0)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        os.makedirs('training_samples', exist_ok=True)
        plt.savefig(f'training_samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

def get_cifar10_dataloader(batch_size=64):
    """Setup CIFAR-10 dataset for training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    return dataloader

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_gan(num_epochs=100, batch_size=64, lr=0.0002, save_interval=10):
    """Train the Interesting GAN"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    # Create models
    generator = InterestingGANGenerator().to(device)
    discriminator = InterestingGANDiscriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print(f"📊 Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"📊 Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Setup data
    dataloader = get_cifar10_dataloader(batch_size)
    print(f"📦 Dataset: CIFAR-10, Batches per epoch: {len(dataloader)}")
    
    # Training loop
    generator.train()
    discriminator.train()
    
    # Track losses
    g_losses = []
    d_losses = []
    
    print(f"🎯 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (real_images, _) in enumerate(pbar):
                batch_size_actual = real_images.size(0)
                real_images = real_images.to(device)
                
                # Create labels
                real_labels = torch.ones(batch_size_actual, 1, device=device)
                fake_labels = torch.zeros(batch_size_actual, 1, device=device)
                
                # ===============================
                # Train Discriminator
                # ===============================
                d_optimizer.zero_grad()
                
                # Real images
                real_output = discriminator(real_images)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size_actual, 100, device=device)
                fake_images = generator(noise)
                fake_output = discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # ===============================
                # Train Generator
                # ===============================
                g_optimizer.zero_grad()
                
                # Generator wants to fool discriminator
                fake_output = discriminator(fake_images)
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                g_optimizer.step()
                
                # Track losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss.item():.4f}',
                    'D_Loss': f'{d_loss.item():.4f}',
                    'D_Real': f'{real_output.mean().item():.3f}',
                    'D_Fake': f'{fake_output.mean().item():.3f}'
                })
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}: G_Loss = {avg_g_loss:.4f}, D_Loss = {avg_d_loss:.4f}")
        
        # Generate sample images
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            visualize_samples(generator, epoch + 1, device)
            
            # Save model checkpoints
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, f'checkpoints/gan_checkpoint_epoch_{epoch+1:03d}.pth')
    
    # Save final model
    torch.save(generator.state_dict(), 'trained_interesting_gan_generator.pth')
    torch.save(discriminator.state_dict(), 'trained_interesting_gan_discriminator.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("🎉 Training completed!")
    print(f"📁 Models saved: trained_interesting_gan_generator.pth")
    print(f"📁 Samples saved: training_samples/")
    print(f"📁 Checkpoints saved: checkpoints/")

def download_pretrained():
    """
    Option to download a pre-trained GAN model
    Note: This would need to be implemented with actual model weights
    """
    print("🔄 Pre-trained model download not implemented yet.")
    print("💡 Options for getting trained weights:")
    print("   1. Train from scratch (run this script)")
    print("   2. Use transfer learning from existing DCGAN")
    print("   3. Download from model zoo (if available)")
    
    # Could implement downloading from:
    # - Hugging Face model hub
    # - PyTorch model zoo
    # - Custom pre-trained weights

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Interesting GAN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--download', action='store_true', help='Download pre-trained model')
    
    args = parser.parse_args()
    
    if args.download:
        download_pretrained()
    else:
        print("🎨 Training Interesting GAN on CIFAR-10")
        print("=" * 50)
        train_gan(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )