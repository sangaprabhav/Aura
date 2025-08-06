#!/usr/bin/env python3
"""
Simplified runner script for MedGemma curriculum learning.
Uses configuration from config.py and provides easy-to-use interface.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import *
from curriculum_medgemma_finetune import AuraDatasetProcessor, CurriculumMedGemmaTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the training environment."""
    logger.info("Setting up training environment...")
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"  # Disable wandb if not needed
    
    # Create output directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}-caption").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}-vqa").mkdir(exist_ok=True)
    
    logger.info("Environment setup completed")

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if dataset exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    # Check for required files
    caption_file = dataset_path / "caption.csv"
    vqa_file = dataset_path / "VQA.csv"
    
    if not caption_file.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
    
    if not vqa_file.exists():
        raise FileNotFoundError(f"VQA file not found: {vqa_file}")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Found {gpu_count} GPU(s): {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 20:
            logger.warning("GPU memory might be insufficient. Consider reducing batch size.")
    else:
        logger.warning("No GPU found. Training will be very slow on CPU.")
    
    logger.info("Prerequisites check completed")

def run_curriculum_training():
    """Run the full curriculum training pipeline."""
    try:
        # Setup
        setup_environment()
        check_prerequisites()
        
        # Initialize components
        logger.info("Initializing training components...")
        data_processor = AuraDatasetProcessor(DATASET_PATH)
        trainer = CurriculumMedGemmaTrainer(MODEL_ID, OUTPUT_DIR)
        
        # Setup model
        trainer.setup_model_and_processor()
        
        # Load and process data
        logger.info("Loading and processing data...")
        caption_df, vqa_df = data_processor.load_data()
        caption_clean, vqa_clean = data_processor.clean_and_filter_data()
        
        # Verify images exist
        caption_clean = data_processor.verify_images_exist(caption_clean)
        vqa_clean = data_processor.verify_images_exist(vqa_clean)
        
        # Create datasets
        caption_dataset = data_processor.create_caption_dataset(
            caption_clean, 
            STAGE1_CONFIG["train_size"], 
            STAGE1_CONFIG["val_size"]
        )
        vqa_dataset = data_processor.create_vqa_dataset(
            vqa_clean, 
            STAGE2_CONFIG["train_size"], 
            STAGE2_CONFIG["val_size"]
        )
        
        logger.info("Data processing completed")
        
        # Stage 1: Caption Generation Training
        logger.info("=" * 60)
        logger.info("STARTING STAGE 1: CAPTION GENERATION TRAINING")
        logger.info("=" * 60)
        
        stage1_model_path = trainer.train_stage(
            caption_dataset, 
            "caption", 
            num_epochs=STAGE1_CONFIG["epochs"], 
            learning_rate=STAGE1_CONFIG["learning_rate"]
        )
        
        # Evaluate Stage 1
        logger.info("Evaluating Stage 1 model...")
        trainer.evaluate_model(
            stage1_model_path, 
            caption_dataset["validation"].select(range(EVAL_CONFIG["eval_samples"])), 
            "caption"
        )
        
        # Stage 2: VQA Training
        logger.info("=" * 60)
        logger.info("STARTING STAGE 2: VQA TRAINING")
        logger.info("=" * 60)
        
        stage2_model_path = trainer.train_stage(
            vqa_dataset, 
            "vqa", 
            num_epochs=STAGE2_CONFIG["epochs"], 
            learning_rate=STAGE2_CONFIG["learning_rate"],
            load_from_checkpoint=stage1_model_path
        )
        
        # Evaluate Stage 2
        logger.info("Evaluating Stage 2 model...")
        trainer.evaluate_model(
            stage2_model_path, 
            vqa_dataset["validation"].select(range(EVAL_CONFIG["eval_samples"])), 
            "vqa"
        )
        
        # Final summary
        logger.info("=" * 60)
        logger.info("CURRICULUM LEARNING COMPLETED SUCCESSFULLY!")
        logger.info(f"Stage 1 model: {stage1_model_path}")
        logger.info(f"Final model: {stage2_model_path}")
        logger.info("=" * 60)
        
        return stage1_model_path, stage2_model_path
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

def run_single_stage(stage: str):
    """Run only a single stage of training."""
    if stage not in ["caption", "vqa"]:
        raise ValueError("Stage must be 'caption' or 'vqa'")
    
    logger.info(f"Running single stage training: {stage}")
    
    # Setup
    setup_environment()
    check_prerequisites()
    
    # Initialize components
    data_processor = AuraDatasetProcessor(DATASET_PATH)
    trainer = CurriculumMedGemmaTrainer(MODEL_ID, OUTPUT_DIR)
    trainer.setup_model_and_processor()
    
    # Load data
    caption_df, vqa_df = data_processor.load_data()
    caption_clean, vqa_clean = data_processor.clean_and_filter_data()
    caption_clean = data_processor.verify_images_exist(caption_clean)
    vqa_clean = data_processor.verify_images_exist(vqa_clean)
    
    if stage == "caption":
        dataset = data_processor.create_caption_dataset(
            caption_clean, 
            STAGE1_CONFIG["train_size"], 
            STAGE1_CONFIG["val_size"]
        )
        config = STAGE1_CONFIG
    else:
        dataset = data_processor.create_vqa_dataset(
            vqa_clean, 
            STAGE2_CONFIG["train_size"], 
            STAGE2_CONFIG["val_size"]
        )
        config = STAGE2_CONFIG
    
    # Train
    model_path = trainer.train_stage(
        dataset, 
        stage, 
        num_epochs=config["epochs"], 
        learning_rate=config["learning_rate"]
    )
    
    # Evaluate
    trainer.evaluate_model(
        model_path, 
        dataset["validation"].select(range(EVAL_CONFIG["eval_samples"])), 
        stage
    )
    
    logger.info(f"Single stage training completed: {model_path}")
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MedGemma curriculum learning")
    parser.add_argument(
        "--stage", 
        choices=["full", "caption", "vqa"], 
        default="full",
        help="Training stage to run (default: full curriculum)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to custom config file (optional)"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        # Override default config with custom config
        globals().update({k: v for k, v in custom_config.__dict__.items() if not k.startswith('_')})
    
    # Run training
    if args.stage == "full":
        run_curriculum_training()
    else:
        run_single_stage(args.stage)