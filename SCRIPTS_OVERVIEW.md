# WAG LLM Fine-Tuning Scripts Overview

*Created: November 29, 2025*

This document provides a quick reference for the scripts created to fine-tune a local LLM for retail ad copy generation.

---

## Project Structure

```
WAG/
├── LLM_FINETUNING_PLAN.md           # Full implementation plan document
├── SCRIPTS_OVERVIEW.md              # This file
│
├── WAG History.xlsx                 # Source: 115K campaign records
├── WAG Master Item.xlsx             # Source: 888K products
├── EABs/                            # Campaign templates
├── PDFs/                            # Visual ad samples
│
└── scripts/
    ├── README.md                     # Detailed scripts documentation
    ├── requirements.txt              # Python dependencies
    ├── run_pipeline.py               # Full pipeline runner
    │
    ├── data_prep/
    │   ├── __init__.py
    │   ├── extract_training_data.py  # Extract from WAG History
    │   ├── enrich_with_products.py   # Add product details
    │   └── validate_data.py          # Data quality validation
    │
    ├── training/
    │   ├── __init__.py
    │   ├── config.yaml               # Training configuration
    │   ├── train.py                  # QLoRA fine-tuning
    │   └── evaluate.py               # Model evaluation
    │
    ├── inference/
    │   ├── __init__.py
    │   ├── generate.py               # Ad copy generation
    │   ├── Modelfile                 # Ollama model definition
    │   └── ollama_setup.py           # Ollama deployment helper
    │
    └── utils/
        ├── __init__.py
        └── helpers.py                # Shared utilities
```

---

## Quick Start

### 1. Setup Environment

```bash
# Navigate to scripts directory
cd WAG/scripts

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run everything: data prep + training + evaluation
python run_pipeline.py --all
```

### 3. Or Run Individual Stages

```bash
# Data preparation only
python run_pipeline.py --prep

# Training only (requires prepared data)
python run_pipeline.py --train

# Evaluation only (requires trained model)
python run_pipeline.py --eval

# Setup Ollama deployment
python run_pipeline.py --ollama

# Check prerequisites
python run_pipeline.py --check
```

---

## Script Descriptions

### Data Preparation (`scripts/data_prep/`)

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `extract_training_data.py` | Extracts headline/body copy pairs | WAG History.xlsx | wag_train.jsonl, wag_val.jsonl, wag_test.jsonl |
| `enrich_with_products.py` | Adds product details (brand, description) | Training JSON + Master Item.xlsx | wag_enriched_*.jsonl |
| `validate_data.py` | Validates data quality, generates reports | Training JSON | validation_report.md |

**Example:**
```bash
cd scripts/data_prep

python extract_training_data.py \
    --input "../../WAG History.xlsx" \
    --output "../output/data"
```

### Training (`scripts/training/`)

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `config.yaml` | Training hyperparameters | - | - |
| `train.py` | QLoRA fine-tuning | Training JSONL + config | LoRA adapter |
| `evaluate.py` | Model evaluation | Adapter + test data | Evaluation report |

**Example:**
```bash
cd scripts/training

# Train with default config
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoint-500
```

### Inference (`scripts/inference/`)

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `generate.py` | Generate ad copy | WIC codes, price, offer | Headline + body copy |
| `ollama_setup.py` | Ollama deployment | Trained adapter | Ollama model |
| `Modelfile` | Ollama model definition | - | - |

**Example:**
```bash
cd scripts/inference

# Single generation
python generate.py --wics "691500,691501" --price "$9.99" --offer "BOGO 50%"

# Batch from EAB file
python generate.py --eab "../../EABs/11.30 EAB.xls" --output results.json

# Using Ollama API
python generate.py --wics "691500" --use-ollama
```

---

## Key Configuration (`training/config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `model.name` | mistralai/Mistral-7B-Instruct-v0.2 | Base model to fine-tune |
| `lora.r` | 64 | LoRA rank (higher = more capacity) |
| `lora.lora_alpha` | 128 | LoRA alpha scaling |
| `training.num_train_epochs` | 3 | Number of training epochs |
| `training.per_device_train_batch_size` | 4 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch = 4 × 8 = 32 |
| `training.learning_rate` | 2e-4 | Learning rate |
| `training.max_seq_length` | 512 | Max sequence length |

---

## Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | RTX 3080 (10GB) | RTX 4080 (16GB) | RTX 4090 (24GB) |
| GPU VRAM | 12 GB | 16 GB | 24 GB |
| System RAM | 32 GB | 64 GB | 128 GB |
| Storage | 50 GB SSD | 100 GB NVMe | 200 GB NVMe |
| Training Time | 8-12 hours | 4-6 hours | 2-3 hours |

---

## Output Locations

After running the pipeline, outputs are stored in `scripts/output/`:

```
scripts/output/
├── data/                          # Extracted training data
│   ├── wag_training_data_raw.json
│   ├── wag_train.jsonl
│   ├── wag_val.jsonl
│   ├── wag_test.jsonl
│   └── extraction_stats.json
│
├── data_enriched/                 # Enriched with product details
│   ├── wag_enriched_full.json
│   ├── wag_enriched_train.jsonl
│   ├── wag_enriched_val.jsonl
│   └── wag_enriched_test.jsonl
│
├── models/                        # Trained models
│   └── wag-copywriter/
│       ├── adapter/               # LoRA weights
│       ├── training_config.yaml
│       └── Modelfile
│
├── checkpoints/                   # Training checkpoints
│   └── checkpoint-*/
│
├── reports/                       # Evaluation reports
│   ├── validation_report.md
│   ├── validation_report.json
│   ├── evaluation.md
│   └── evaluation.json
│
└── logs/                          # Training logs
```

---

## Ollama Deployment

### Quick Setup (No Training Required)

Get started immediately with the base Mistral model:

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

# List available models
python ollama_setup.py --list
```

### API Usage

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Brand: Advil | Price: $8.99 | Offer: BOGO 50%",
  "stream": false
}'
```

---

## Troubleshooting

### Out of Memory (OOM)

```yaml
# In config.yaml, reduce batch size:
training:
  per_device_train_batch_size: 2  # Reduced from 4
  gradient_accumulation_steps: 16  # Increased to maintain effective batch
```

### Slow Training

- Install Unsloth for 2x faster training: `pip install unsloth`
- Enable `group_by_length: true` in config
- Increase `dataloader_num_workers`

### Poor Generation Quality

- Increase training epochs (try 5-6)
- Use enriched data with product details
- Try higher LoRA rank (r: 128)
- Adjust temperature (0.5 for more focused, 0.9 for more creative)

### Data Preparation Issues

```bash
# If Excel files fail to load, ensure these are installed:
pip install openpyxl xlrd
```

---

## Next Steps After Setup

1. **Review the plan:** Read `LLM_FINETUNING_PLAN.md` for full details
2. **Check prerequisites:** `python run_pipeline.py --check`
3. **Prepare data:** `python run_pipeline.py --prep`
4. **Review validation:** Check `scripts/output/reports/validation_report.md`
5. **Train model:** `python run_pipeline.py --train`
6. **Evaluate:** `python run_pipeline.py --eval`
7. **Deploy:** `python run_pipeline.py --ollama`
8. **Test generation:** `python scripts/inference/generate.py --wics "691500"`

---

## Support

For questions or issues, contact the Enterprise Architecture Team.

---

*Document auto-generated with project setup*
