#!/usr/bin/env python3
"""
Test script to verify the setup and data loading.
Run this before starting the full training to catch issues early.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠ CUDA not available - training will be slow")
            
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    required_packages = [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("bitsandbytes", "BitsAndBytes"),
        ("PIL", "Pillow"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            return False
    
    return True

def test_dataset():
    """Test if dataset files exist and are readable."""
    print("\nTesting dataset...")
    
    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        print("✗ Dataset directory not found")
        return False
    
    # Check CSV files
    caption_file = dataset_path / "caption.csv"
    vqa_file = dataset_path / "VQA.csv"
    
    if not caption_file.exists():
        print("✗ caption.csv not found")
        return False
    
    if not vqa_file.exists():
        print("✗ VQA.csv not found")
        return False
    
    # Try to read CSV files
    try:
        import pandas as pd
        
        caption_df = pd.read_csv(caption_file)
        print(f"✓ Caption data: {len(caption_df)} samples")
        
        vqa_df = pd.read_csv(vqa_file)
        print(f"✓ VQA data: {len(vqa_df)} samples")
        
        # Check for required columns
        if 'image' not in caption_df.columns or 'caption' not in caption_df.columns:
            print("✗ Caption CSV missing required columns")
            return False
            
        if 'image' not in vqa_df.columns or 'question' not in vqa_df.columns or 'answer' not in vqa_df.columns:
            print("✗ VQA CSV missing required columns")
            return False
        
        # Check if some images exist
        sample_images = caption_df['image'].head(5)
        existing_images = 0
        for img_path in sample_images:
            full_path = dataset_path / img_path
            if full_path.exists():
                existing_images += 1
        
        if existing_images == 0:
            print("✗ No sample images found")
            return False
        else:
            print(f"✓ Found {existing_images}/5 sample images")
            
    except Exception as e:
        print(f"✗ Error reading dataset: {e}")
        return False
    
    return True

def test_model_access():
    """Test if MedGemma model can be accessed."""
    print("\nTesting model access...")
    
    try:
        from transformers import AutoProcessor
        
        # Try to load processor (lightweight test)
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        print("✓ MedGemma model accessible")
        return True
        
    except Exception as e:
        print(f"✗ MedGemma access failed: {e}")
        print("  Make sure you have:")
        print("  1. Accepted the license at https://huggingface.co/google/medgemma-4b-it")
        print("  2. Set HF_TOKEN environment variable or run 'huggingface-cli login'")
        return False

def test_memory():
    """Test available system memory."""
    print("\nTesting system resources...")
    
    try:
        import psutil
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / 1e9:.1f}GB ({memory.percent}% used)")
        
        if memory.available < 16e9:  # Less than 16GB available
            print("⚠ Low system memory - consider closing other applications")
        
        # Disk space
        disk = psutil.disk_usage('.')
        print(f"✓ Disk space: {disk.free / 1e9:.1f}GB available")
        
        if disk.free < 50e9:  # Less than 50GB available
            print("⚠ Low disk space - training may fail")
            
    except ImportError:
        print("⚠ psutil not available - skipping memory check")
    except Exception as e:
        print(f"⚠ Memory check failed: {e}")

def main():
    """Run all tests."""
    print("MedGemma Curriculum Learning Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test dataset
    if not test_dataset():
        all_passed = False
    
    # Test model access
    if not test_model_access():
        all_passed = False
    
    # Test system resources
    test_memory()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Ready to start training.")
        print("\nTo start training, run:")
        print("python run_curriculum_training.py")
    else:
        print("✗ Some tests failed. Please fix the issues before training.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())