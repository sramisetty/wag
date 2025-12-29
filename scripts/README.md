# WAG Ad Copy LLM Fine-Tuning Scripts

Scripts for fine-tuning a local LLM to generate retail advertising headlines and body copy for Walgreens campaigns.

## Directory Structure

```
scripts/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── data_prep/                   # Data preparation scripts
│   ├── extract_training_data.py # Extract from WAG History
│   ├── enrich_with_products.py  # Add product details from Master Item
│   └── validate_data.py         # Validate data quality
│
├── training/                    # Model training scripts
│   ├── config.yaml              # Training configuration
│   ├── train.py                 # Main training script (QLoRA)
│   └── evaluate.py              # Model evaluation
│
├── inference/                   # Generation & deployment
│   ├── generate.py              # Generate ad copy
│   ├── Modelfile                # Ollama model definition
│   └── ollama_setup.py          # Ollama deployment helper
│
├── utils/                       # Utility modules
│   └── (shared utilities)
│
└── output/                      # Generated outputs
    ├── data/                    # Extracted training data
    ├── data_enriched/           # Enriched training data
    ├── models/                  # Fine-tuned models
    ├── reports/                 # Evaluation reports
    └── logs/                    # Training logs
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
cd scripts/data_prep

# Step 1: Extract training pairs from WAG History
python extract_training_data.py \
    --input "../../WAG History.xlsx" \
    --output "../output/data"

# Step 2: Enrich with product details (optional but recommended)
python enrich_with_products.py \
    --training "../output/data/wag_training_data_raw.json" \
    --master "../../WAG Master Item.xlsx" \
    --output "../output/data_enriched"

# Step 3: Validate data quality
python validate_data.py \
    --input "../output/data/wag_training_data_raw.json" \
    --output "../output/reports/validation_report.md"
```

### 3. Train the Model

```bash
cd scripts/training

# Run training with default config
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoint-500
```

### 4. Evaluate the Model

```bash
cd scripts/training

python evaluate.py \
    --model ../output/models/wag-copywriter \
    --test ../output/data/wag_test.jsonl \
    --output ../output/reports/evaluation
```

### 5. Generate Ad Copy

```bash
cd scripts/inference

# Single generation
python generate.py \
    --wics "691500,691501" \
    --price "$9.99" \
    --offer "BOGO 50%"

# Batch from EAB file
python generate.py \
    --eab "../../EABs/11.30 EAB.xls" \
    --output generated_copy.json

# Using Ollama (after setup)
python generate.py \
    --wics "691500" \
    --use-ollama
```

## Configuration

### Training Configuration (`training/config.yaml`)

Key settings to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `model.name` | mistralai/Mistral-7B-Instruct-v0.2 | Base model |
| `lora.r` | 64 | LoRA rank (higher = more capacity) |
| `training.num_train_epochs` | 3 | Number of training epochs |
| `training.per_device_train_batch_size` | 4 | Batch size per GPU |
| `training.learning_rate` | 2e-4 | Learning rate |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 24 GB |
| System RAM | 32 GB | 64 GB |
| Storage | 50 GB | 100 GB |

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WAG History.xlsx ─────► extract_training_data.py               │
│        │                         │                               │
│        │                         ▼                               │
│        │               wag_train.jsonl                           │
│        │               wag_val.jsonl                             │
│        │               wag_test.jsonl                            │
│        │                         │                               │
│        ▼                         ▼                               │
│  WAG Master Item.xlsx ──► enrich_with_products.py               │
│                                  │                               │
│                                  ▼                               │
│                         wag_enriched_*.jsonl                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  config.yaml + training data ──► train.py                       │
│                                      │                           │
│                                      ▼                           │
│  Mistral 7B + QLoRA ──────► Fine-tuned Adapter                  │
│                                      │                           │
│                                      ▼                           │
│                              evaluate.py                         │
│                                      │                           │
│                                      ▼                           │
│                           Evaluation Report                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option A: Python Inference                                      │
│  ─────────────────────────                                       │
│  generate.py --model ./adapter --wics "..."                      │
│                                                                  │
│  Option B: Ollama API                                            │
│  ────────────────────                                            │
│  1. ollama_setup.py --setup-base  (immediate, no training)      │
│  2. ollama_setup.py --create      (after training)              │
│  3. ollama run wag-copywriter                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Output Files

### Data Preparation

| File | Description |
|------|-------------|
| `wag_training_data_raw.json` | All extracted examples (JSON) |
| `wag_train.jsonl` | Training split (80%) |
| `wag_val.jsonl` | Validation split (10%) |
| `wag_test.jsonl` | Test split (10%) |
| `extraction_stats.json` | Data statistics |
| `validation_report.md` | Data quality report |

### Training

| File | Description |
|------|-------------|
| `output/models/wag-copywriter/adapter/` | LoRA adapter weights |
| `output/models/wag-copywriter/training_config.yaml` | Training config used |
| `output/models/wag-copywriter/Modelfile` | Ollama model definition |
| `output/checkpoints/checkpoint-*/` | Training checkpoints |

### Evaluation

| File | Description |
|------|-------------|
| `evaluation.json` | Detailed evaluation results |
| `evaluation.md` | Human-readable report |

## Using Ollama

### Quick Setup (No Training Required)

For immediate use with the base Mistral model:

```bash
cd scripts/inference

# Setup base model with WAG system prompt
python ollama_setup.py --setup-base

# Test it
ollama run wag-copywriter-base
```

### After Fine-Tuning

```bash
# Create model from fine-tuned adapter
python ollama_setup.py --create

# Test the model
python ollama_setup.py --test

# Use via API
curl http://localhost:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Price: $8.99 | Offer: BOGO 50%",
  "stream": false
}'
```

## Tips & Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps` proportionally
- Enable `gradient_checkpointing: true`
- Use a smaller LoRA rank (`r: 32` instead of `64`)

### Slow Training

- Use Unsloth for 2x faster training: `pip install unsloth`
- Enable `group_by_length: true`
- Increase `dataloader_num_workers`

### Poor Generation Quality

- Increase training epochs
- Try higher LoRA rank
- Use enriched data (with product details)
- Adjust temperature (0.5-0.9)

### Ollama Issues

- Ensure Ollama is running: `ollama serve`
- Check model exists: `ollama list`
- View logs: `ollama logs`

## File Descriptions

### Data Preparation Scripts

#### `extract_training_data.py`
Extracts headline/body copy pairs from WAG History.xlsx.
- Parses 115K+ campaign records
- Extracts WIC codes, pricing, offer types
- Creates train/val/test splits (80/10/10)
- Outputs instruction-format JSONL

#### `enrich_with_products.py`
Enriches training data with product details from Master Item.
- Joins WIC codes to product descriptions
- Adds brand, vendor, category information
- Improves model context for generation

#### `validate_data.py`
Validates data quality and generates reports.
- Checks for missing/invalid fields
- Analyzes patterns in headlines/body copy
- Computes data statistics
- Identifies duplicates

### Training Scripts

#### `train.py`
Main training script using QLoRA.
- Loads base model with 4-bit quantization
- Applies LoRA adapters
- Runs supervised fine-tuning
- Saves checkpoints and final model

#### `evaluate.py`
Evaluates model performance.
- Generates on test set
- Computes exact/partial match scores
- Creates detailed evaluation report

### Inference Scripts

#### `generate.py`
Generates ad copy using the trained model.
- Single generation mode
- Batch processing from EAB files
- Local model or Ollama API support

#### `ollama_setup.py`
Helper for Ollama deployment.
- Creates Ollama models from adapters
- Sets up base model for immediate use
- Tests model functionality

## License

For internal use only. Based on open-source models with permissive licenses (Apache 2.0).

## Support

Contact: Enterprise Architecture Team
