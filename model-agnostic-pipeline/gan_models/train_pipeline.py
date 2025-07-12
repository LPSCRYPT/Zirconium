#!/usr/bin/env python3
"""
Complete GAN Training Pipeline
Organized training system for Compact GAN with dataset management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from gan_compact import CompactGANGenerator, CompactGANDiscriminator

class CustomImageDataset(Dataset):
    """
    Custom dataset loader for user-provided images
    Supports common image formats in datasets/ directory
    """
    
    def __init__(self, dataset_path, transform=None, image_size=16):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_paths.extend([
                os.path.join(root, file)
                for root, dirs, files in os.walk(dataset_path)
                for file in files if file.lower().endswith(ext)
            ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        print(f"📊 Found {len(self.image_paths)} images in dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and convert to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, 0  # Return dummy label
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.image_size, self.image_size), 0

class GANTrainingPipeline:
    """
    Complete GAN training pipeline with logging, checkpointing, and visualization
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Training on device: {self.device}")
        
        # Create models
        self.generator = CompactGANGenerator().to(self.device)
        self.discriminator = CompactGANDiscriminator().to(self.device)
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
        # Setup optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config['lr'], 
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config['lr'], 
            betas=(0.5, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training tracking
        self.g_losses = []
        self.d_losses = []
        self.epoch = 0
        
        # Setup directories
        self.checkpoint_dir = os.path.join('gan_models', 'checkpoints', config['experiment_name'])
        self.log_dir = os.path.join('gan_models', 'logs', config['experiment_name'])
        self.sample_dir = os.path.join('gan_models', 'outputs', config['experiment_name'], 'samples')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        print(f"📁 Experiment: {config['experiment_name']}")
        print(f"📁 Checkpoints: {self.checkpoint_dir}")
        print(f"📁 Logs: {self.log_dir}")
        print(f"📁 Samples: {self.sample_dir}")
    
    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def setup_dataset(self, dataset_path):
        """Setup dataset from user-provided images"""
        
        # Define transforms for 16x16 RGB
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])
        
        # Create dataset
        dataset = CustomImageDataset(dataset_path, transform=transform)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        print(f"📦 Dataset: {len(dataset)} images")
        print(f"📦 Batches per epoch: {len(dataloader)}")
        
        return dataloader
    
    def generate_samples(self, num_samples=16, save_path=None):
        """Generate and save sample images"""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate samples
            noise = torch.randn(num_samples, 50, device=self.device)
            fake_images = self.generator(noise)
            
            # Convert from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
            # Create grid
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            fig.suptitle(f'Generated Samples - Epoch {self.epoch}', fontsize=14)
            
            for i in range(num_samples):
                row, col = i // 4, i % 4
                # Convert from CHW to HWC for matplotlib
                img = fake_images[i].cpu().permute(1, 2, 0)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                axes[row, col].set_title(f'#{i+1}', fontsize=8)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.sample_dir, f'epoch_{self.epoch:03d}.png')
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Saved samples: {save_path}")
        
        self.generator.train()
    
    def save_checkpoint(self, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.epoch:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if specified
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💫 Saved best checkpoint: {best_path}")
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        print(f"✅ Loaded checkpoint from epoch {self.epoch}")
        return True
    
    def log_metrics(self, g_loss, d_loss, d_real_acc, d_fake_acc):
        """Log training metrics"""
        log_data = {
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'generator_loss': g_loss,
            'discriminator_loss': d_loss,
            'discriminator_real_accuracy': d_real_acc,
            'discriminator_fake_accuracy': d_fake_acc,
        }
        
        # Save to JSON log
        log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def plot_losses(self):
        """Plot and save loss curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss', color='blue')
        plt.plot(self.d_losses, label='Discriminator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average for smoother curves
        if len(self.g_losses) > 10:
            window = min(10, len(self.g_losses) // 10)
            g_smooth = [sum(self.g_losses[max(0, i-window):i+1]) / min(i+1, window) 
                       for i in range(len(self.g_losses))]
            d_smooth = [sum(self.d_losses[max(0, i-window):i+1]) / min(i+1, window) 
                       for i in range(len(self.d_losses))]
            plt.plot(g_smooth, label='Generator (smooth)', color='lightblue')
            plt.plot(d_smooth, label='Discriminator (smooth)', color='lightcoral')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_real_acc = 0
        epoch_d_fake_acc = 0
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f'Epoch {self.epoch+1}') as pbar:
            for batch_idx, (real_images, _) in enumerate(pbar):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # ===============================
                # Train Discriminator
                # ===============================
                self.d_optimizer.zero_grad()
                
                # Real images
                real_output = self.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, 50, device=self.device)
                fake_images = self.generator(noise)
                fake_output = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Calculate accuracies
                d_real_acc = (real_output > 0.5).float().mean().item()
                d_fake_acc = (fake_output < 0.5).float().mean().item()
                
                # ===============================
                # Train Generator
                # ===============================
                self.g_optimizer.zero_grad()
                
                # Generator wants to fool discriminator
                fake_output = self.discriminator(fake_images)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Track losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss.item():.4f}',
                    'D_Loss': f'{d_loss.item():.4f}',
                    'D_Real': f'{d_real_acc:.3f}',
                    'D_Fake': f'{d_fake_acc:.3f}'
                })
        
        # Calculate average metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_d_real_acc = epoch_d_real_acc / num_batches
        avg_d_fake_acc = epoch_d_fake_acc / num_batches
        
        return avg_g_loss, avg_d_loss, avg_d_real_acc, avg_d_fake_acc
    
    def train(self, dataset_path, resume_from=None):
        """Main training loop"""
        print(f"🎯 Starting GAN Training")
        print(f"{'='*50}")
        
        # Setup dataset
        dataloader = self.setup_dataset(dataset_path)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Print model info
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"📊 Generator parameters: {g_params:,}")
        print(f"📊 Discriminator parameters: {d_params:,}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            # Train one epoch
            g_loss, d_loss, d_real_acc, d_fake_acc = self.train_epoch(dataloader)
            
            # Track losses
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            # Log metrics
            self.log_metrics(g_loss, d_loss, d_real_acc, d_fake_acc)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}: "
                  f"G_Loss = {g_loss:.4f}, D_Loss = {d_loss:.4f}, "
                  f"D_Real_Acc = {d_real_acc:.3f}, D_Fake_Acc = {d_fake_acc:.3f}")
            
            # Generate samples
            if (epoch + 1) % self.config['sample_interval'] == 0:
                self.generate_samples()
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                is_best = len(self.g_losses) > 5 and g_loss == min(self.g_losses[-5:])
                self.save_checkpoint(is_best=is_best)
            
            # Plot losses
            if (epoch + 1) % self.config['plot_interval'] == 0:
                self.plot_losses()
        
        # Final saves
        self.generate_samples()
        self.save_checkpoint(is_best=True)
        self.plot_losses()
        
        print(f"🎉 Training completed!")
        print(f"📁 Results saved in: gan_models/")

def create_training_config(experiment_name="gan_16x16_custom"):
    """Create default training configuration"""
    return {
        'experiment_name': experiment_name,
        'num_epochs': 100,
        'batch_size': 32,
        'lr': 0.0002,
        'sample_interval': 5,      # Generate samples every N epochs
        'checkpoint_interval': 10, # Save checkpoint every N epochs
        'plot_interval': 10,       # Update plots every N epochs
    }

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Compact GAN')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to dataset directory (e.g., gan_models/datasets/my_images)')
    parser.add_argument('--experiment', type=str, default='gan_16x16_custom',
                       help='Experiment name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Create config
    config = create_training_config(args.experiment)
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    
    # Create pipeline
    pipeline = GANTrainingPipeline(config)
    
    # Start training
    pipeline.train(args.dataset, resume_from=args.resume)

if __name__ == "__main__":
    main()