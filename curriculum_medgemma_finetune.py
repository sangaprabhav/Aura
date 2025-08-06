#!/usr/bin/env python3
"""
Curriculum Learning Fine-tuning for MedGemma on Aura Dataset

This script implements a two-stage curriculum learning approach:
1. Stage 1: Image captioning (easier task)
2. Stage 2: Visual Question Answering (harder task)

Based on Google's MedGemma fine-tuning example with adaptations for curriculum learning.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import logging

# Hugging Face imports
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    BitsAndBytesConfig,
    pipeline
)
from datasets import Dataset, DatasetDict, load_metric
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuraDatasetProcessor:
    """Processes the Aura dataset for curriculum learning."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.caption_df = None
        self.vqa_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load caption and VQA datasets."""
        logger.info("Loading Aura dataset...")
        
        # Load caption data
        caption_path = self.dataset_path / "caption.csv"
        self.caption_df = pd.read_csv(caption_path)
        logger.info(f"Loaded {len(self.caption_df)} caption samples")
        
        # Load VQA data
        vqa_path = self.dataset_path / "VQA.csv"
        self.vqa_df = pd.read_csv(vqa_path)
        logger.info(f"Loaded {len(self.vqa_df)} VQA samples")
        
        return self.caption_df, self.vqa_df
    
    def clean_and_filter_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean and filter the datasets."""
        logger.info("Cleaning and filtering data...")
        
        # Filter caption data - remove empty captions
        caption_clean = self.caption_df.dropna(subset=['caption'])
        caption_clean = caption_clean[caption_clean['caption'].str.strip() != '']
        
        # Filter VQA data - remove empty questions/answers
        vqa_clean = self.vqa_df.dropna(subset=['question', 'answer'])
        vqa_clean = vqa_clean[
            (vqa_clean['question'].str.strip() != '') & 
            (vqa_clean['answer'].str.strip() != '')
        ]
        
        logger.info(f"After cleaning: {len(caption_clean)} caption samples, {len(vqa_clean)} VQA samples")
        
        return caption_clean, vqa_clean
    
    def verify_images_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify that image files exist."""
        logger.info("Verifying image files exist...")
        
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = self.dataset_path / row['image']
            if image_path.exists():
                valid_rows.append(row)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        result_df = pd.DataFrame(valid_rows)
        logger.info(f"Found {len(result_df)} samples with valid images")
        
        return result_df
    
    def create_caption_dataset(self, df: pd.DataFrame, train_size: int = 8000, val_size: int = 1000) -> DatasetDict:
        """Create dataset for caption generation task."""
        logger.info("Creating caption dataset...")
        
        # Sample data if needed
        if len(df) > train_size + val_size:
            df_sampled = df.sample(n=train_size + val_size, random_state=42)
        else:
            df_sampled = df
        
        # Split into train/val
        train_df = df_sampled.iloc[:train_size]
        val_df = df_sampled.iloc[train_size:train_size + val_size]
        
        def format_caption_data(row):
            """Format data for caption generation."""
            image_path = self.dataset_path / row['image']
            
            # Use cleaned_caption if available, otherwise use caption
            caption_text = row.get('cleaned_caption', row['caption'])
            if pd.isna(caption_text) or caption_text.strip() == '':
                caption_text = row['caption']
            
            return {
                'image': str(image_path),
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': 'Describe this medical image in detail.'}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': caption_text}
                        ]
                    }
                ]
            }
        
        train_data = [format_caption_data(row) for _, row in train_df.iterrows()]
        val_data = [format_caption_data(row) for _, row in val_df.iterrows()]
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })
    
    def create_vqa_dataset(self, df: pd.DataFrame, train_size: int = 8000, val_size: int = 1000) -> DatasetDict:
        """Create dataset for VQA task."""
        logger.info("Creating VQA dataset...")
        
        # Sample data if needed
        if len(df) > train_size + val_size:
            df_sampled = df.sample(n=train_size + val_size, random_state=42)
        else:
            df_sampled = df
        
        # Split into train/val
        train_df = df_sampled.iloc[:train_size]
        val_df = df_sampled.iloc[train_size:train_size + val_size]
        
        def format_vqa_data(row):
            """Format data for VQA."""
            image_path = self.dataset_path / row['image']
            
            return {
                'image': str(image_path),
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': row['question']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': row['answer']}
                        ]
                    }
                ]
            }
        
        train_data = [format_vqa_data(row) for _, row in train_df.iterrows()]
        val_data = [format_vqa_data(row) for _, row in val_df.iterrows()]
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })


class CurriculumMedGemmaTrainer:
    """Curriculum learning trainer for MedGemma."""
    
    def __init__(self, model_id: str = "google/medgemma-4b-it", output_dir: str = "medgemma-curriculum"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.processor = None
        self.model = None
        
        # Check GPU capability
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] < 8:
                logger.warning("GPU does not support bfloat16, using float16 instead")
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.bfloat16
        else:
            logger.warning("No GPU available, using CPU")
            self.torch_dtype = torch.float32
    
    def setup_model_and_processor(self):
        """Initialize model and processor."""
        logger.info(f"Loading model: {self.model_id}")
        
        # Model configuration
        model_kwargs = {
            "attn_implementation": "eager",
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        # Add quantization if using GPU
        if torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_storage=self.torch_dtype,
            )
        
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "right"
        
        logger.info("Model and processor loaded successfully")
    
    def create_data_collator(self):
        """Create data collator for multimodal data."""
        def collate_fn(examples: List[Dict[str, Any]]):
            texts = []
            images = []
            
            for example in examples:
                # Load image
                image_path = example["image"]
                image = Image.open(image_path).convert("RGB")
                images.append([image])
                
                # Process text
                text = self.processor.apply_chat_template(
                    example["messages"], 
                    add_generation_prompt=False, 
                    tokenize=False
                ).strip()
                texts.append(text)
            
            # Tokenize and process
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            
            # Create labels
            labels = batch["input_ids"].clone()
            
            # Mask special tokens
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.tokenizer.special_tokens_map["boi_token"]
            )
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            labels[labels == 262144] = -100  # Additional special token
            
            batch["labels"] = labels
            return batch
        
        return collate_fn
    
    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration."""
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )
    
    def train_stage(self, dataset: DatasetDict, stage_name: str, num_epochs: int = 2, 
                   learning_rate: float = 2e-4, load_from_checkpoint: Optional[str] = None) -> str:
        """Train a single stage of curriculum learning."""
        logger.info(f"Starting {stage_name} training...")
        
        # Load from checkpoint if specified
        if load_from_checkpoint:
            logger.info(f"Loading model from checkpoint: {load_from_checkpoint}")
            self.model = AutoModelForImageTextToText.from_pretrained(load_from_checkpoint)
        
        # Training configuration
        stage_output_dir = f"{self.output_dir}-{stage_name}"
        
        args = SFTConfig(
            output_dir=stage_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # Reduced for memory
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="steps",
            eval_steps=100,
            learning_rate=learning_rate,
            bf16=self.torch_dtype == torch.bfloat16,
            fp16=self.torch_dtype == torch.float16,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            push_to_hub=False,  # Set to True if you want to push to Hub
            report_to="tensorboard",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"].shuffle().select(range(min(200, len(dataset["validation"])))),
            peft_config=self.get_lora_config(),
            processing_class=self.processor,
            data_collator=self.create_data_collator(),
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        logger.info(f"{stage_name} training completed. Model saved to {stage_output_dir}")
        
        # Clean up
        del trainer
        torch.cuda.empty_cache()
        
        return stage_output_dir
    
    def evaluate_model(self, model_path: str, test_dataset: Dataset, task_type: str = "caption"):
        """Evaluate the trained model."""
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model for evaluation
        eval_pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=self.processor,
            torch_dtype=self.torch_dtype,
        )
        
        eval_pipe.model.generation_config.do_sample = False
        eval_pipe.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.processor.tokenizer.padding_side = "left"
        
        # Run evaluation
        test_messages = []
        test_images = []
        references = []
        
        for example in test_dataset:
            test_messages.append(example["messages"])
            test_images.append(Image.open(example["image"]).convert("RGB"))
            # Extract reference text from assistant message
            references.append(example["messages"][1]["content"][0]["text"])
        
        # Generate predictions
        outputs = eval_pipe(
            text=test_messages,
            images=test_images,
            max_new_tokens=150 if task_type == "caption" else 100,
            batch_size=16,
            return_full_text=False,
        )
        
        predictions = [out[0]["generated_text"] for out in outputs]
        
        # Compute BLEU score as a basic metric
        try:
            bleu_metric = evaluate.load("bleu")
            bleu_score = bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            logger.info(f"BLEU Score: {bleu_score['bleu']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute BLEU score: {e}")
        
        # Clean up
        del eval_pipe
        torch.cuda.empty_cache()
        
        return predictions, references


def main():
    """Main training function."""
    # Configuration
    DATASET_PATH = "/Users/prabhavsanga/Desktop/Aura/Dataset"
    OUTPUT_DIR = "medgemma-aura-curriculum"
    
    # Stage 1 parameters
    STAGE1_EPOCHS = 2
    STAGE1_LR = 2e-4
    STAGE1_TRAIN_SIZE = 6000
    STAGE1_VAL_SIZE = 800
    
    # Stage 2 parameters  
    STAGE2_EPOCHS = 2
    STAGE2_LR = 1e-4  # Lower learning rate for second stage
    STAGE2_TRAIN_SIZE = 6000
    STAGE2_VAL_SIZE = 800
    
    try:
        # Initialize components
        logger.info("Initializing curriculum learning pipeline...")
        data_processor = AuraDatasetProcessor(DATASET_PATH)
        trainer = CurriculumMedGemmaTrainer(output_dir=OUTPUT_DIR)
        
        # Setup model
        trainer.setup_model_and_processor()
        
        # Load and process data
        caption_df, vqa_df = data_processor.load_data()
        caption_clean, vqa_clean = data_processor.clean_and_filter_data()
        
        # Verify images exist
        caption_clean = data_processor.verify_images_exist(caption_clean)
        vqa_clean = data_processor.verify_images_exist(vqa_clean)
        
        # Create datasets
        caption_dataset = data_processor.create_caption_dataset(
            caption_clean, STAGE1_TRAIN_SIZE, STAGE1_VAL_SIZE
        )
        vqa_dataset = data_processor.create_vqa_dataset(
            vqa_clean, STAGE2_TRAIN_SIZE, STAGE2_VAL_SIZE
        )
        
        logger.info("Dataset preparation completed")
        
        # Stage 1: Caption Generation Training
        logger.info("=" * 50)
        logger.info("STAGE 1: Caption Generation Training")
        logger.info("=" * 50)
        
        stage1_model_path = trainer.train_stage(
            caption_dataset, 
            "caption", 
            num_epochs=STAGE1_EPOCHS, 
            learning_rate=STAGE1_LR
        )
        
        # Evaluate Stage 1
        logger.info("Evaluating Stage 1 model...")
        trainer.evaluate_model(
            stage1_model_path, 
            caption_dataset["validation"].select(range(100)), 
            "caption"
        )
        
        # Stage 2: VQA Training (starting from Stage 1 model)
        logger.info("=" * 50)
        logger.info("STAGE 2: VQA Training")
        logger.info("=" * 50)
        
        stage2_model_path = trainer.train_stage(
            vqa_dataset, 
            "vqa", 
            num_epochs=STAGE2_EPOCHS, 
            learning_rate=STAGE2_LR,
            load_from_checkpoint=stage1_model_path
        )
        
        # Evaluate Stage 2
        logger.info("Evaluating Stage 2 model...")
        trainer.evaluate_model(
            stage2_model_path, 
            vqa_dataset["validation"].select(range(100)), 
            "vqa"
        )
        
        logger.info("=" * 50)
        logger.info("CURRICULUM LEARNING COMPLETED!")
        logger.info(f"Final model saved at: {stage2_model_path}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()