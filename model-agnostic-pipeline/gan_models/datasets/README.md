# 📁 GAN Training Datasets

## Upload Your Images Here

### Supported Formats
- **Image types**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Resolution**: Any size (automatically resized to 16×16)
- **Color**: RGB or grayscale (converted to RGB)

### Directory Structure
```
datasets/
├── my_images/           # Your custom dataset
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── faces/               # Face dataset
├── objects/             # Object dataset
└── art/                 # Art/texture dataset
```

### Dataset Requirements
- **Minimum**: 100+ images for basic training
- **Recommended**: 1000+ images for good results
- **Optimal**: 10000+ images for high quality

### Example Datasets You Can Use
1. **CIFAR-10**: Download with the training script
2. **Custom faces**: Upload portrait photos
3. **Art/textures**: Upload artistic images
4. **Objects**: Upload object photos
5. **Logos**: Upload logo images

### How to Add Your Dataset
1. Create folder: `mkdir gan_models/datasets/your_dataset_name`
2. Copy images: `cp /path/to/your/images/* gan_models/datasets/your_dataset_name/`
3. Start training: `python gan_models/train_pipeline.py --dataset gan_models/datasets/your_dataset_name`

### Training Commands

**Basic Training:**
```bash
python gan_models/train_pipeline.py --dataset gan_models/datasets/my_images
```

**Custom Training:**
```bash
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/my_images \
  --experiment my_custom_gan \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0002
```

**Resume Training:**
```bash
python gan_models/train_pipeline.py \
  --dataset gan_models/datasets/my_images \
  --resume gan_models/checkpoints/my_custom_gan/latest.pth
```

The training pipeline will automatically:
- ✅ Resize images to 16×16
- ✅ Convert to RGB format
- ✅ Normalize pixel values
- ✅ Create data batches
- ✅ Save training samples
- ✅ Log training metrics
- ✅ Save model checkpoints