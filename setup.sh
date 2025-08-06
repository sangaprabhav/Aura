#!/bin/bash

# MedGemma Curriculum Learning Setup Script
echo "Setting up MedGemma Curriculum Learning Environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check Python version (should be 3.8+)
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Check if Hugging Face token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo "Or run: huggingface-cli login"
fi

# Check dataset
if [ ! -f "Dataset/caption.csv" ] || [ ! -f "Dataset/VQA.csv" ]; then
    echo "Warning: Dataset files not found in Dataset/ directory"
    echo "Please ensure caption.csv and VQA.csv are present"
fi

# Create output directories
mkdir -p logs
mkdir -p models

echo "Setup completed!"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "python run_curriculum_training.py"
echo ""
echo "For help, run:"
echo "python run_curriculum_training.py --help"