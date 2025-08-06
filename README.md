# MedGemma Curriculum Learning Fine-tuning

This project implements curriculum learning for fine-tuning Google's MedGemma model on the Aura medical image dataset. The approach uses a two-stage training process:

1. **Stage 1**: Image captioning (easier task)
2. **Stage 2**: Visual Question Answering (harder task)

## Dataset Overview

The Aura dataset contains:
- **11,002 medical images** (dermatology and pathology)
- **11,091 image captions** from medical textbooks
- **27,262 VQA pairs** for medical visual question answering
- Images sourced from 15 different medical publications
- Coverage of skin conditions, pathology, and dermoscopy

## Quick Start

### 1. Setup Environment

```bash
# Clone/download the project and navigate to directory
cd /path/to/Aura

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Configure Hugging Face Access

You need access to MedGemma on Hugging Face:

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Request access to [MedGemma](https://huggingface.co/google/medgemma-4b-it)
3. Create a [write token](https://huggingface.co/settings/tokens)
4. Set your token:

```bash
export HF_TOKEN=your_token_here
# OR
huggingface-cli login
```

### 3. Run Training

**Full curriculum training:**
```bash
python run_curriculum_training.py
```

**Single stage training:**
```bash
# Caption generation only
python run_curriculum_training.py --stage caption

# VQA only
python run_curriculum_training.py --stage vqa
```

## Project Structure

```
Aura/
├── Dataset/                          # Aura dataset
│   ├── caption.csv                   # Image captions
│   ├── VQA.csv                      # Question-answer pairs
│   └── *.png                        # Medical images
├── curriculum_medgemma_finetune.py  # Main training implementation
├── run_curriculum_training.py       # Easy-to-use runner script
├── config.py                        # Configuration parameters
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Environment setup script
└── README.md                        # This file
```

## Configuration

Edit `config.py` to customize training parameters:

### Key Parameters

```python
# Model
MODEL_ID = "google/medgemma-4b-it"

# Stage 1: Caption Generation
STAGE1_CONFIG = {
    "epochs": 2,
    "learning_rate": 2e-4,
    "train_size": 6000,
    "val_size": 800,
}

# Stage 2: VQA
STAGE2_CONFIG = {
    "epochs": 2,
    "learning_rate": 1e-4,  # Lower for second stage
    "train_size": 6000,
    "val_size": 800,
}
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space
- **CUDA**: Compatible CUDA installation

### Recommended Requirements
- **GPU**: NVIDIA A100 (40GB) or H100
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+ NVMe SSD

### Memory Optimization

The script uses several memory optimization techniques:
- **4-bit quantization** with BitsAndBytesConfig
- **LoRA** (Low-Rank Adaptation) for parameter-efficient training
- **Gradient checkpointing** to reduce memory usage
- **Small batch sizes** with gradient accumulation

## Training Process

### Stage 1: Caption Generation
- **Task**: Generate medical image descriptions
- **Duration**: ~2-3 hours on A100
- **Purpose**: Learn basic visual-text alignment

### Stage 2: VQA Fine-tuning
- **Task**: Answer questions about medical images
- **Duration**: ~2-3 hours on A100
- **Purpose**: Learn complex reasoning and medical knowledge

### Curriculum Learning Benefits
1. **Improved convergence**: Easier task first helps model learn basic patterns
2. **Better performance**: Sequential learning often outperforms joint training
3. **Reduced overfitting**: Gradual complexity increase improves generalization

## Monitoring Training

### TensorBoard
```bash
# In another terminal
tensorboard --logdir=medgemma-aura-curriculum-caption/logs
tensorboard --logdir=medgemma-aura-curriculum-vqa/logs
```

### Training Logs
- Console output saved to `training.log`
- Model checkpoints saved to `medgemma-aura-curriculum-*/`
- Evaluation metrics logged during training

## Evaluation

The script automatically evaluates models using:
- **BLEU score** for text generation quality
- **Sample predictions** on validation set
- **Loss tracking** during training

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size in config.py
STAGE1_CONFIG["batch_size"] = 1
STAGE2_CONFIG["batch_size"] = 1

# Increase gradient accumulation
STAGE1_CONFIG["gradient_accumulation_steps"] = 16
```

**2. Slow Training**
- Ensure you're using GPU: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Consider using smaller dataset sizes for testing

**3. Dataset Issues**
- Verify image paths in CSV files match actual files
- Check image file formats (should be readable by PIL)
- Ensure CSV files are properly formatted

**4. Model Access Issues**
- Verify HF token has correct permissions
- Ensure you've accepted MedGemma license terms
- Check internet connection for model download

### Performance Tips

1. **Use mixed precision training** (enabled by default)
2. **Enable gradient checkpointing** for memory efficiency
3. **Use appropriate batch sizes** based on your GPU
4. **Monitor GPU memory usage** with `nvidia-smi`

## Advanced Usage

### Custom Configuration

Create a custom config file:

```python
# my_config.py
from config import *

# Override specific parameters
STAGE1_CONFIG["epochs"] = 3
STAGE2_CONFIG["learning_rate"] = 5e-5
```

Run with custom config:
```bash
python run_curriculum_training.py --config my_config.py
```

### Resuming Training

To resume from a checkpoint:
```python
# In curriculum_medgemma_finetune.py
trainer.train(resume_from_checkpoint="path/to/checkpoint")
```

## Model Output

After training, you'll have:
- **Stage 1 model**: `medgemma-aura-curriculum-caption/`
- **Final model**: `medgemma-aura-curriculum-vqa/`

Both models can be loaded with Hugging Face Transformers:

```python
from transformers import pipeline

# Load the final model
pipe = pipeline(
    "image-text-to-text",
    model="medgemma-aura-curriculum-vqa",
    torch_dtype=torch.bfloat16
)

# Use for inference
result = pipe(image=image, text="What condition is shown in this image?")
```

## Citation

If you use this code or the Aura dataset, please cite:

```bibtex
@software{medgemma_curriculum_2024,
  title={MedGemma Curriculum Learning Fine-tuning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/medgemma-curriculum}
}
```

## License

This project is licensed under the Apache License 2.0. See the original MedGemma license terms for model usage.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the training logs for error messages
3. Ensure your hardware meets the requirements
4. Verify dataset integrity and format