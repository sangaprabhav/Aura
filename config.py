"""
Configuration file for MedGemma curriculum learning fine-tuning.
Adjust these parameters based on your hardware and requirements.
"""

# Model Configuration
MODEL_ID = "google/medgemma-4b-it"
OUTPUT_DIR = "medgemma-aura-curriculum"

# Dataset Configuration
DATASET_PATH = "/Users/prabhavsanga/Desktop/Aura/Dataset"

# Stage 1: Caption Generation
STAGE1_CONFIG = {
    "epochs": 2,
    "learning_rate": 2e-4,
    "train_size": 6000,
    "val_size": 800,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_new_tokens": 150,
}

# Stage 2: Visual Question Answering
STAGE2_CONFIG = {
    "epochs": 2,
    "learning_rate": 1e-4,  # Lower LR for second stage
    "train_size": 6000,
    "val_size": 800,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_new_tokens": 100,
}

# LoRA Configuration
LORA_CONFIG = {
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "r": 16,
    "bias": "none",
    "target_modules": "all-linear",
}

# Training Configuration
TRAINING_CONFIG = {
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "logging_steps": 50,
    "save_strategy": "epoch",
    "eval_strategy": "steps",
    "eval_steps": 100,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "linear",
    "push_to_hub": False,  # Set to True if you want to push to Hugging Face Hub
    "report_to": "tensorboard",
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_quantization": True,  # Use 4-bit quantization
    "use_gradient_checkpointing": True,
    "max_memory_per_gpu": "40GB",  # Adjust based on your GPU
}

# Evaluation Configuration
EVAL_CONFIG = {
    "eval_batch_size": 16,
    "eval_samples": 100,  # Number of samples to evaluate on
    "compute_bleu": True,
    "compute_rouge": False,  # Set to True if you want ROUGE scores
}