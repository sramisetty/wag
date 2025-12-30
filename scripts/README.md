# WAG Ad Copy LLM Fine-Tuning Pipeline

A complete solution for fine-tuning a local LLM (Mistral 7B) to generate retail advertising headlines and body copy for Walgreens campaigns.

**Version:** 1.0.0
**Last Updated:** December 2025

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Setup Guide](#setup-guide)
- [API Server](#api-server)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Inference & Generation](#inference--generation)
- [Ollama Deployment](#ollama-deployment)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

---

## Overview

This project provides an end-to-end pipeline for:

1. **Data Extraction** - Extract 115K+ campaign records from WAG History
2. **Data Enrichment** - Enhance with 888K+ product details from Master Item
3. **Model Training** - Fine-tune Mistral 7B using QLoRA (4-bit quantization)
4. **Evaluation** - Measure quality with BLEU, ROUGE, and exact match metrics
5. **Deployment** - Serve via REST API or Ollama

### Technology Stack

| Component | Technology |
|-----------|------------|
| Base Model | Mistral 7B Instruct v0.2 |
| Fine-Tuning | QLoRA (4-bit NF4 quantization) |
| Framework | PyTorch + Hugging Face Transformers |
| API Server | Flask with CORS |
| Local LLM | Ollama |
| Language | Python 3.10+ |

---

## Project Structure

```
/mnt/data/sri/wag/
├── WAG History.xlsx              # Training data: 115K campaign records
├── WAG Master Item.xlsx          # Product master: 888K products
├── EABs/                         # Campaign template files (11 files)
├── PDFs/                         # Ad samples by market (20 PDFs)
├── LLM_FINETUNING_PLAN.md        # Strategic planning document
├── SCRIPTS_OVERVIEW.md           # Quick reference guide
│
└── scripts/
    ├── README.md                 # This file
    ├── requirements.txt          # Python dependencies
    ├── run_pipeline.py           # Master orchestration script
    ├── venv/                     # Python virtual environment
    │
    ├── data_prep/                # Data preparation pipeline
    │   ├── extract_training_data.py
    │   ├── enrich_with_products.py
    │   └── validate_data.py
    │
    ├── training/                 # Model training
    │   ├── config.yaml           # Training configuration
    │   ├── train.py              # QLoRA fine-tuning script
    │   └── evaluate.py           # Model evaluation
    │
    ├── inference/                # Generation & deployment
    │   ├── api_server.py         # Flask REST API server
    │   ├── generate.py           # CLI generation tool
    │   ├── ollama_setup.py       # Ollama deployment helper
    │   ├── Modelfile             # Ollama model definition
    │   ├── Modelfile.base        # Base model (no training)
    │   ├── start.sh              # Start API server
    │   ├── stop.sh               # Stop API server
    │   └── restart.sh            # Restart API server
    │
    ├── utils/                    # Shared utilities
    │   └── helpers.py
    │
    └── output/                   # Generated outputs
        ├── data/                 # Extracted training data
        ├── data_enriched/        # Enriched training data
        ├── models/               # Fine-tuned models
        ├── reports/              # Validation & evaluation reports
        ├── checkpoints/          # Training checkpoints
        └── logs/                 # Training logs
```

---

## Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | RTX 3080 (10GB) | RTX 4080 (16GB) | RTX 4090 (24GB) |
| GPU VRAM | 12 GB | 16 GB | 24 GB |
| System RAM | 32 GB | 64 GB | 128 GB |
| Storage | 50 GB SSD | 100 GB NVMe | 200 GB NVMe |
| Training Time | 8-12 hours | 4-6 hours | 2-3 hours |

### Software Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- Ollama (for local LLM deployment)
- Git

---

## Quick Start

### 1. Activate Virtual Environment

```bash
cd /mnt/data/sri/wag/scripts
source venv/bin/activate
```

### 2. Start API Server (for immediate use)

```bash
cd inference
./start.sh
```

Access the API at: `http://<your-ip>:5000/`

### 3. Run Full Pipeline (for training)

```bash
python run_pipeline.py --all
```

---

## Setup Guide

### Step 1: Clone/Access the Project

```bash
cd /mnt/data/sri/wag/scripts
```

### Step 2: Create Virtual Environment (if not exists)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python run_pipeline.py --check
```

This verifies:
- Python packages are installed
- GPU/CUDA is available
- Source data files exist
- Ollama is running (if installed)

---

## API Server

The Flask API server provides REST endpoints for ad copy generation.

### Starting the Server

```bash
cd /mnt/data/sri/wag/scripts/inference

# Start server (runs in background)
./start.sh

# Or with custom port
PORT=8080 ./start.sh

# For foreground (debugging)
source ../venv/bin/activate
python api_server.py --host 0.0.0.0 --port 5000
```

### Stopping the Server

```bash
./stop.sh
```

### Restarting the Server

```bash
./restart.sh
```

### Network Access

To access the API from other machines on the network:

1. Start server with `--host 0.0.0.0` (default in start.sh)
2. Access via: `http://<server-ip>:5000/`

Example: `http://172.16.1.130:5000/`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI with interactive testing |
| GET | `/api/health` | Health check and status |
| GET | `/api/models` | List available models |
| GET | `/api/training/status` | Get fine-tuning progress |
| POST | `/api/generate` | Generate ad copy for products |
| POST | `/api/generate/batch` | Batch generation for multiple items |
| POST | `/api/generate/eab` | Process EAB slot data |

### Web UI Features

The web interface at `/` includes:

- **Model Selector** - Dropdown to choose from available Ollama models
- **Temperature Control** - Adjustable creativity setting (0-2)
- **Training Status Panel** - Real-time fine-tuning progress with:
  - Progress bar with percentage complete
  - Current step / Total steps
  - Current epoch / Total epochs
  - Loss value
  - Elapsed time and ETA
  - Auto-refreshes every 5 seconds

### API Examples

#### Health Check

```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "ollama": "healthy",
  "model": "wag-copywriter",
  "version": "1.0.0"
}
```

#### Generate Ad Copy

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "products": [{"description": "ADVIL PM 20CT", "brand": "Advil"}],
    "price": "$8.99",
    "offer": "BOGO 50%"
  }'
```

Response:
```json
{
  "headline": "Advil Pain Relief",
  "body_copy": "Select varieties. Limit 2.",
  "model": "wag-copywriter",
  "elapsed_ms": 1234
}
```

#### Batch Generation

```bash
curl -X POST http://localhost:5000/api/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"products": [{"description": "ADVIL PM 20CT"}], "price": "$8.99"},
      {"products": [{"description": "TYLENOL 100CT"}], "price": "$12.99"}
    ]
  }'
```

#### Training Status

```bash
curl http://localhost:5000/api/training/status
```

Response (during training):
```json
{
  "status": "training",
  "phase": "in_progress",
  "current_step": 500,
  "total_steps": 2000,
  "current_epoch": 1.25,
  "total_epochs": 3,
  "percent_complete": 25.0,
  "loss": 0.4523,
  "elapsed_formatted": "01:30:45",
  "eta_formatted": "04:32:15",
  "message": "Training step 500..."
}
```

### Server Logs

Logs are written to: `/mnt/data/sri/wag/scripts/inference/api_server.log`

```bash
tail -f api_server.log
```

---

## Data Preparation

### Step 1: Extract Training Data

Extract headline/body copy pairs from WAG History:

```bash
cd /mnt/data/sri/wag/scripts/data_prep

python extract_training_data.py \
    --input "../../WAG History.xlsx" \
    --output "../output/data"
```

**Output:**
- `wag_training_data_raw.json` - All extracted examples
- `wag_train.jsonl` - Training split (80%)
- `wag_val.jsonl` - Validation split (10%)
- `wag_test.jsonl` - Test split (10%)
- `extraction_stats.json` - Statistics

### Step 2: Enrich with Product Details (Recommended)

Add product descriptions, brands, and categories:

```bash
python enrich_with_products.py \
    --training "../output/data/wag_training_data_raw.json" \
    --master "../../WAG Master Item.xlsx" \
    --output "../output/data_enriched"
```

### Step 3: Validate Data Quality

Generate quality report:

```bash
python validate_data.py \
    --input "../output/data/wag_training_data_raw.json" \
    --output "../output/reports/validation_report.md"
```

**Current Statistics:**
- Total Records: 112,728
- Training Suitable: 97.5%
- With Body Copy: 72.8%
- Unique Products: 25,835

---

## Model Training

### Basic Training

```bash
cd /mnt/data/sri/wag/scripts/training

python train.py --config config.yaml
```

### Resume from Checkpoint

```bash
python train.py --config config.yaml --resume checkpoint-500
```

### Training Output

Models are saved to: `output/models/wag-copywriter/`
- `adapter/` - LoRA adapter weights
- `training_config.yaml` - Configuration used
- `Modelfile` - Ollama model definition

### Evaluation

```bash
python evaluate.py \
    --model ../output/models/wag-copywriter \
    --test ../output/data/wag_test.jsonl \
    --output ../output/reports/evaluation
```

---

## Inference & Generation

### Command Line Generation

```bash
cd /mnt/data/sri/wag/scripts/inference

# Single generation
python generate.py \
    --wics "691500,691501" \
    --price "$9.99" \
    --offer "BOGO 50%"

# Batch from EAB file
python generate.py \
    --eab "../../EABs/11.30 EAB.xls" \
    --output generated_copy.json

# Using Ollama
python generate.py \
    --wics "691500" \
    --use-ollama
```

---

## Ollama Deployment

### Quick Setup (No Training Required)

Use the base Mistral model with WAG system prompt:

```bash
cd /mnt/data/sri/wag/scripts/inference

# Setup base model
python ollama_setup.py --setup-base

# Test it
ollama run wag-copywriter-base
```

### After Fine-Tuning

```bash
# Create model from trained adapter
python ollama_setup.py --create

# Test the model
python ollama_setup.py --test

# List models
python ollama_setup.py --list
```

### Ollama API Usage

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Brand: Advil | Price: $8.99 | Offer: BOGO 50%",
  "stream": false
}'
```

---

## Configuration Reference

### Training Configuration (`training/config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `model.name` | mistralai/Mistral-7B-Instruct-v0.2 | Base model |
| `quantization.bits` | 4 | Quantization bits (4 or 8) |
| `lora.r` | 64 | LoRA rank (higher = more capacity) |
| `lora.lora_alpha` | 128 | LoRA alpha scaling |
| `lora.lora_dropout` | 0.05 | Dropout rate |
| `training.num_train_epochs` | 3 | Number of epochs |
| `training.per_device_train_batch_size` | 4 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch = 4 × 8 = 32 |
| `training.learning_rate` | 2e-4 | Learning rate |
| `training.max_seq_length` | 512 | Maximum sequence length |
| `training.warmup_ratio` | 0.03 | Warmup ratio |
| `training.lr_scheduler_type` | cosine | LR scheduler |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | http://localhost:11434 | Ollama API URL |
| `WAG_MODEL` | wag-copywriter | Default model name |
| `REQUEST_TIMEOUT` | 60 | API request timeout (seconds) |
| `HOST` | 0.0.0.0 | API server host |
| `PORT` | 5000 | API server port |

---

## Troubleshooting

### API Server Issues

**Port already in use:**
```bash
# Kill existing process
sudo fuser -k 5000/tcp
# Or use different port
PORT=5001 ./start.sh
```

**Cannot access from network:**
- Ensure server started with `--host 0.0.0.0`
- Check firewall: `sudo ufw status`
- Verify port is open: `sudo ufw allow 5000`

**404 on root URL:**
- Restart the server completely
- Clear Python cache: `rm -rf __pycache__`

### Training Issues

**Out of Memory (OOM):**
```yaml
# In config.yaml:
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch
  gradient_checkpointing: true  # Enable checkpointing
```

**Slow Training:**
- Install Unsloth: `pip install unsloth`
- Enable `group_by_length: true`
- Increase `dataloader_num_workers`

**Poor Generation Quality:**
- Increase training epochs (5-6)
- Use enriched data with product details
- Try higher LoRA rank (r: 128)
- Adjust temperature (0.5 for focused, 0.9 for creative)

### Ollama Issues

**Ollama not responding:**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

**Model not found:**
```bash
# List available models
ollama list

# Pull base model
ollama pull mistral:7b-instruct
```

### Data Issues

**Excel files won't load:**
```bash
pip install openpyxl xlrd
```

**Virtual environment issues:**
```bash
# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Pipeline Commands Reference

```bash
# Full pipeline
python run_pipeline.py --all

# Individual stages
python run_pipeline.py --prep      # Data preparation
python run_pipeline.py --train     # Training
python run_pipeline.py --eval      # Evaluation
python run_pipeline.py --ollama    # Ollama setup

# Utilities
python run_pipeline.py --check     # Verify prerequisites
```

---

## Output Files Reference

### Data Outputs (`output/data/`)

| File | Description |
|------|-------------|
| `wag_training_data_raw.json` | All extracted examples |
| `wag_train.jsonl` | Training split (80%) |
| `wag_val.jsonl` | Validation split (10%) |
| `wag_test.jsonl` | Test split (10%) |
| `extraction_stats.json` | Extraction statistics |

### Enriched Data (`output/data_enriched/`)

| File | Description |
|------|-------------|
| `wag_enriched_full.json` | All enriched examples |
| `wag_enriched_train.jsonl` | Enriched training split |
| `wag_enriched_val.jsonl` | Enriched validation split |
| `wag_enriched_test.jsonl` | Enriched test split |

### Reports (`output/reports/`)

| File | Description |
|------|-------------|
| `validation_report.md` | Data quality report |
| `validation_report.json` | Structured validation data |
| `evaluation.md` | Model evaluation report |
| `evaluation.json` | Detailed evaluation metrics |

### Models (`output/models/wag-copywriter/`)

| File | Description |
|------|-------------|
| `adapter/` | LoRA adapter weights |
| `training_config.yaml` | Training configuration used |
| `Modelfile` | Ollama model definition |

---

## Support

**Team:** Enterprise Architecture Team

**Documentation:**
- `LLM_FINETUNING_PLAN.md` - Project planning and strategy
- `MODEL_GUIDE.md` - Model architecture and post-training roadmap
- `SCRIPTS_OVERVIEW.md` - Quick reference guide

---

## License

For internal use only. Based on open-source models with permissive licenses (Apache 2.0).
