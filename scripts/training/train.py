"""
WAG Ad Copy Fine-Tuning Training Script
========================================
Fine-tunes Mistral 7B (or other LLMs) using QLoRA for ad copy generation

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoint-500

Requirements:
    pip install torch transformers peft bitsandbytes accelerate trl datasets

Author: Enterprise Architecture Team
Created: November 2025
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class WAGTrainer:
    """Fine-tuning trainer for WAG ad copy generation."""

    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    def setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration for QLoRA."""
        if not self.config['quantization']['enabled']:
            return None

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(
                torch,
                self.config['quantization']['bnb_4bit_compute_dtype']
            ),
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
        )

        logger.info("Configured 4-bit quantization (QLoRA)")
        return quant_config

    def load_model_and_tokenizer(self) -> None:
        """Load the base model and tokenizer."""
        model_name = self.config['model']['name']
        logger.info(f"Loading model: {model_name}")

        # Setup quantization
        quant_config = self.setup_quantization()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config['model'].get('trust_remote_code', True),
            padding_side="right",
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=self.config['model'].get('device_map', 'auto'),
            trust_remote_code=self.config['model'].get('trust_remote_code', True),
            torch_dtype=getattr(torch, self.config['model'].get('torch_dtype', 'bfloat16')),
        )

        # Prepare model for k-bit training
        if quant_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True)
            )

        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")

    def setup_lora(self) -> None:
        """Configure and apply LoRA to the model."""
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type'],
            target_modules=self.config['lora']['target_modules'],
        )

        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"LoRA configured with r={self.config['lora']['r']}, alpha={self.config['lora']['lora_alpha']}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def load_dataset(self) -> tuple:
        """Load training and validation datasets."""
        data_config = self.config['data']

        # Try enriched data first, fall back to basic
        train_path = data_config['train_file']
        val_path = data_config['val_file']

        if not Path(train_path).exists():
            train_path = data_config.get('train_file_fallback', train_path)
            val_path = data_config.get('val_file_fallback', val_path)

        logger.info(f"Loading training data from: {train_path}")
        logger.info(f"Loading validation data from: {val_path}")

        # Load JSONL files
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))

        val_data = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line))

        # Limit samples if specified
        max_samples = data_config.get('max_samples')
        if max_samples:
            train_data = train_data[:max_samples]
            val_data = val_data[:min(max_samples // 10, len(val_data))]

        logger.info(f"Training samples: {len(train_data):,}")
        logger.info(f"Validation samples: {len(val_data):,}")

        # Convert to Dataset
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        return train_dataset, val_dataset

    def format_prompt(self, example: Dict) -> str:
        """Format a single example into a prompt."""
        template = self.config['data']['prompt_template']

        return template.format(
            instruction=example.get('instruction', ''),
            input=example.get('input', ''),
            output=example.get('output', '')
        )

    def preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess examples for training."""
        prompts = []
        for i in range(len(examples['instruction'])):
            example = {
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i]
            }
            prompts.append(self.format_prompt(example))

        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.config['training']['max_seq_length'],
            padding='max_length',
            return_tensors=None,
        )

        tokenized['labels'] = tokenized['input_ids'].copy()

        return tokenized

    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments."""
        train_config = self.config['training']

        # Create output directory
        output_dir = Path(train_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=train_config['num_train_epochs'],
            max_steps=train_config.get('max_steps', -1),
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            learning_rate=train_config['learning_rate'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            warmup_ratio=train_config['warmup_ratio'],
            weight_decay=train_config['weight_decay'],
            optim=train_config['optim'],
            fp16=train_config.get('fp16', False),
            bf16=train_config.get('bf16', True),
            gradient_checkpointing=train_config.get('gradient_checkpointing', True),
            logging_steps=train_config['logging_steps'],
            save_strategy=train_config['save_strategy'],
            save_steps=train_config['save_steps'],
            save_total_limit=train_config['save_total_limit'],
            evaluation_strategy=train_config['eval_strategy'],
            eval_steps=train_config['eval_steps'],
            seed=train_config['seed'],
            dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
            remove_unused_columns=train_config.get('remove_unused_columns', True),
            group_by_length=train_config.get('group_by_length', True),
            report_to=train_config.get('report_to', 'none'),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        logger.info(f"Training arguments configured")
        logger.info(f"  - Epochs: {args.num_train_epochs}")
        logger.info(f"  - Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"  - Learning rate: {args.learning_rate}")

        return args

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run the training loop."""
        logger.info("=" * 60)
        logger.info("Starting WAG Ad Copy Fine-Tuning")
        logger.info("=" * 60)

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Setup LoRA
        self.setup_lora()

        # Load datasets
        train_dataset, val_dataset = self.load_dataset()

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )

        # Setup training arguments
        training_args = self.setup_training_args()

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            max_seq_length=self.config['training']['max_seq_length'],
        )

        # Train
        logger.info("Starting training...")
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.trainer.train(resume_from_checkpoint=resume_from)
        else:
            self.trainer.train()

        # Save final model
        self.save_model()

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

    def save_model(self) -> None:
        """Save the fine-tuned model and adapter."""
        output_dir = Path(self.config['training']['output_dir'])

        # Save adapter
        adapter_dir = output_dir / "adapter"
        self.model.save_pretrained(adapter_dir)
        logger.info(f"Saved LoRA adapter to: {adapter_dir}")

        # Save tokenizer
        self.tokenizer.save_pretrained(adapter_dir)
        logger.info(f"Saved tokenizer to: {adapter_dir}")

        # Save training config
        config_path = output_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved training config to: {config_path}")

        # Create Modelfile for Ollama
        self.create_ollama_modelfile(output_dir)

    def create_ollama_modelfile(self, output_dir: Path) -> None:
        """Create Ollama Modelfile for deployment."""
        modelfile_content = self.config['ollama']['modelfile_template']

        modelfile_path = output_dir / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        logger.info(f"Created Ollama Modelfile: {modelfile_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune LLM for WAG ad copy generation'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available! Training will be very slow.")

    # Run training
    trainer = WAGTrainer(args.config)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
