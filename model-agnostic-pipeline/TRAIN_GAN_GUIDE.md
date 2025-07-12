# 🎨 Complete GAN Training Guide

## 🚀 Quick Start

### Step 1: Prepare Your Dataset
```bash
# Create dataset directory
mkdir -p gan_models/datasets/my_images

# Copy your images (replace with your image path)
cp /path/to/your/images/* gan_models/datasets/my_images/

# Verify images were copied
ls gan_models/datasets/my_images/
```

### Step 2: Start Training
```bash
# Basic training (recommended settings)
python gan_models/train_pipeline.py --dataset gan_models/datasets/my_images

# Custom training with your preferences
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/my_images \
  --experiment my_gan_experiment \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.0001
```

### Step 3: Monitor Training
```bash
# Watch generated samples (updated every 5 epochs)
ls gan_models/outputs/my_gan_experiment/samples/

# Check training logs
tail -f gan_models/logs/my_gan_experiment/training_log.jsonl

# View loss curves
open gan_models/logs/my_gan_experiment/loss_curves.png
```

## 📊 Training Options

### Command Line Arguments
```bash
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/YOUR_FOLDER \     # Required: path to images
  --experiment EXPERIMENT_NAME \                  # Optional: experiment name
  --epochs 100 \                                  # Optional: training epochs
  --batch-size 32 \                              # Optional: batch size
  --lr 0.0002 \                                  # Optional: learning rate
  --resume PATH_TO_CHECKPOINT                     # Optional: resume training
```

### Dataset Requirements
- **Format**: JPG, PNG, BMP, TIFF images
- **Size**: Any resolution (auto-resized to 16×16)
- **Quantity**: 
  - Minimum: 100 images
  - Good: 1000+ images  
  - Excellent: 5000+ images

## 🔧 Training Configuration

### Hardware Requirements
- **RAM**: 2-4GB minimum, 8GB+ recommended
- **GPU**: Optional but speeds up training 10x
- **Storage**: 1GB+ for checkpoints and samples
- **Time**: 30min - 4 hours depending on dataset size

### Default Settings (Optimized)
```python
config = {
    'experiment_name': 'gan_16x16_custom',
    'num_epochs': 100,
    'batch_size': 32,
    'lr': 0.0002,
    'sample_interval': 5,      # Generate samples every 5 epochs
    'checkpoint_interval': 10, # Save checkpoint every 10 epochs
    'plot_interval': 10,       # Update plots every 10 epochs
}
```

## 📈 Training Monitoring

### Generated Samples
Training automatically generates sample images:
```
gan_models/outputs/YOUR_EXPERIMENT/samples/
├── epoch_005.png    # Samples after 5 epochs
├── epoch_010.png    # Samples after 10 epochs
└── ...
```

### Training Logs
Detailed JSON logs for analysis:
```
gan_models/logs/YOUR_EXPERIMENT/
├── training_log.jsonl    # Per-epoch metrics
└── loss_curves.png       # Training loss visualization
```

### Checkpoints
Automatic checkpoint saving:
```
gan_models/checkpoints/YOUR_EXPERIMENT/
├── latest.pth           # Most recent checkpoint
├── best.pth            # Best performing checkpoint
└── checkpoint_epoch_XXX.pth  # Regular checkpoints
```

## 🎯 Training Tips

### For Best Results:
1. **Curate your dataset**: Remove low-quality/corrupted images
2. **Consistent theme**: Similar style images train better
3. **Sufficient data**: 1000+ images recommended
4. **Monitor early**: Check samples after 10-20 epochs
5. **Adjust learning rate**: Lower if training unstable

### Common Issues:
- **Mode collapse**: All generated images look the same
  - Solution: Lower learning rate, increase dataset diversity
- **Training instability**: Losses oscillate wildly
  - Solution: Reduce learning rate to 0.0001
- **Poor quality**: Blurry or random images
  - Solution: Train longer, improve dataset quality

## 🔄 Using Trained Models

### Export for EZKL Verification
After training, use the trained weights:

```python
# Update gan_compact.py to load trained weights
class TrainedCompactGAN(CompactGANModel):
    def get_model(self):
        if self._model is None:
            self._model = CompactGANGenerator()
            # Load your trained weights
            self._model.load_state_dict(torch.load('gan_models/checkpoints/YOUR_EXPERIMENT/best.pth')['generator_state_dict'])
            self._model.eval()
        return self._model
```

### Generate ZK Proofs with Trained Model
```bash
# Register trained model and run EZKL
python demo.py --model trained_gan_compact --blockchain-scripts
```

## 📝 Example Workflows

### Workflow 1: Face Generation
```bash
# 1. Collect face images
mkdir gan_models/datasets/faces
# ... add face images ...

# 2. Train face GAN
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/faces \
  --experiment face_gan \
  --epochs 100

# 3. Monitor results
ls gan_models/outputs/face_gan/samples/
```

### Workflow 2: Logo Generation  
```bash
# 1. Collect logo images
mkdir gan_models/datasets/logos
# ... add logo images ...

# 2. Train logo GAN
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/logos \
  --experiment logo_gan \
  --epochs 200 \
  --batch-size 64

# 3. Use for verification
# ... integrate trained weights and verify on blockchain ...
```

## 🎉 Success Metrics

### Training Complete When:
- ✅ Generated samples look realistic
- ✅ Diverse outputs (not mode collapse)
- ✅ Stable training losses
- ✅ Satisfactory visual quality

### Next Steps:
1. **Integrate trained weights** into EZKL model
2. **Generate zero-knowledge proofs** 
3. **Deploy on blockchain** for verification
4. **Create verifiable NFTs** or other applications

The training pipeline provides everything needed to go from raw images to blockchain-verified generative AI!