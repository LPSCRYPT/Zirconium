#!/usr/bin/env python3
"""
Quick training test - just 3 epochs to verify everything works
"""

import subprocess
import sys
import os

def run_quick_training_test():
    """Run a quick 3-epoch training test"""
    print("🧪 Running Quick GAN Training Test")
    print("=" * 50)
    
    # Training command
    cmd = [
        sys.executable, 
        "gan_models/train_pipeline.py",
        "--dataset", "gan_models/datasets/sample_training",
        "--experiment", "quick_test",
        "--epochs", "3",
        "--batch-size", "16"  # Small batch for quick test
    ]
    
    print("🚀 Command:", " ".join(cmd))
    print("⏳ This will take about 1-2 minutes...")
    print()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        
        print("\n🎉 Quick training test SUCCESSFUL!")
        
        # Check outputs
        experiment_dir = "gan_models/outputs/quick_test"
        if os.path.exists(experiment_dir):
            samples_dir = os.path.join(experiment_dir, "samples")
            if os.path.exists(samples_dir):
                samples = os.listdir(samples_dir)
                print(f"📁 Generated samples: {samples}")
        
        # Check checkpoints
        checkpoint_dir = "gan_models/checkpoints/quick_test"
        if os.path.exists(checkpoint_dir):
            checkpoints = os.listdir(checkpoint_dir)
            print(f"💾 Saved checkpoints: {checkpoints}")
        
        # Check logs
        log_dir = "gan_models/logs/quick_test"
        if os.path.exists(log_dir):
            logs = os.listdir(log_dir)
            print(f"📊 Training logs: {logs}")
        
        print("\n✅ Training pipeline is working correctly!")
        print("🎯 Ready for full training with your dataset!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training test failed: {e}")
        print("💡 Check the error messages above")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 60)
    print("🎯 Next Steps: Train with Your Dataset")
    print("=" * 60)
    
    print("\n1️⃣ **Upload Your Images**:")
    print("   mkdir gan_models/datasets/my_images")
    print("   cp /path/to/your/images/* gan_models/datasets/my_images/")
    
    print("\n2️⃣ **Start Full Training**:")
    print("   python gan_models/train_pipeline.py \\")
    print("     --dataset gan_models/datasets/my_images \\")
    print("     --experiment my_gan \\")
    print("     --epochs 50")
    
    print("\n3️⃣ **Monitor Progress**:")
    print("   # Watch generated samples:")
    print("   ls gan_models/outputs/my_gan/samples/")
    print("   # Check training logs:")
    print("   tail -f gan_models/logs/my_gan/training_log.jsonl")
    
    print("\n4️⃣ **Use Trained Model for EZKL**:")
    print("   # After training completes, integrate weights and generate proofs")
    print("   # See TRAIN_GAN_GUIDE.md for details")
    
    print("\n💡 **Tips**:")
    print("   - Use 1000+ images for best results")
    print("   - Train for 50-100 epochs")
    print("   - Monitor early samples (epoch 10-20)")
    print("   - Adjust learning rate if training unstable")

if __name__ == "__main__":
    success = run_quick_training_test()
    show_next_steps()
    
    if success:
        print("\n🎉 GAN Training Pipeline Setup Complete!")
    else:
        print("\n❌ Setup needs debugging. Check error messages above.")