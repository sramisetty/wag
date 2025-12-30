# WAG LLM Fine-Tuning Scripts Overview

*Last Updated: December 2025*

Quick reference for the WAG Ad Copy LLM fine-tuning project. For detailed documentation, see `scripts/README.md`.

---

## Project Structure

```
/mnt/data/sri/wag/
├── WAG History.xlsx                 # Source: 115K campaign records
├── WAG Master Item.xlsx             # Source: 888K products
├── EABs/                            # 11 campaign template files
├── PDFs/                            # 20 visual ad samples (by market)
├── LLM_FINETUNING_PLAN.md           # Strategic planning document
├── SCRIPTS_OVERVIEW.md              # This file
│
└── scripts/
    ├── README.md                    # Detailed documentation
    ├── requirements.txt             # Python dependencies
    ├── run_pipeline.py              # Master orchestration script
    ├── venv/                        # Virtual environment
    │
    ├── data_prep/
    │   ├── extract_training_data.py # Extract from WAG History
    │   ├── enrich_with_products.py  # Add product details
    │   └── validate_data.py         # Data quality validation
    │
    ├── training/
    │   ├── config.yaml              # Training configuration
    │   ├── train.py                 # QLoRA fine-tuning
    │   └── evaluate.py              # Model evaluation
    │
    ├── inference/
    │   ├── api_server.py            # Flask REST API server
    │   ├── generate.py              # CLI generation tool
    │   ├── ollama_setup.py          # Ollama deployment helper
    │   ├── Modelfile                # Ollama model definition
    │   ├── Modelfile.base           # Base model (no training)
    │   ├── start.sh                 # Start API server
    │   ├── stop.sh                  # Stop API server
    │   └── restart.sh               # Restart API server
    │
    ├── utils/
    │   └── helpers.py               # Shared utilities
    │
    └── output/
        ├── data/                    # Extracted training data
        ├── data_enriched/           # Enriched training data
        ├── models/                  # Fine-tuned models
        ├── reports/                 # Validation/evaluation reports
        ├── checkpoints/             # Training checkpoints
        └── logs/                    # Training logs
```

---

## Quick Start

### 1. Activate Virtual Environment

```bash
cd /mnt/data/sri/wag/scripts
source venv/bin/activate
```

### 2. Start API Server

```bash
cd inference
./start.sh
```

Access: `http://<server-ip>:5000/`

### 3. Run Full Pipeline

```bash
python run_pipeline.py --all
```

---

## API Server Management

### Start/Stop/Restart

```bash
cd /mnt/data/sri/wag/scripts/inference

./start.sh           # Start in background
./stop.sh            # Stop server
./restart.sh         # Restart server

# Custom port
PORT=8080 ./start.sh

# Foreground mode (debugging)
python api_server.py --host 0.0.0.0 --port 5000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI with interactive testing |
| GET | `/api/health` | Health check |
| GET | `/api/models` | List available models |
| GET | `/api/training/status` | Fine-tuning progress |
| POST | `/api/generate` | Generate ad copy |
| POST | `/api/generate/batch` | Batch generation |
| POST | `/api/generate/eab` | Process EAB data |

### Web UI Features

- **Model Selector** - Choose from available Ollama models
- **Temperature Control** - Adjust creativity (0-2)
- **Training Status Panel** - Real-time progress, loss, ETA (auto-refreshes)

### Network Access

Access from other machines: `http://<server-ip>:5000/`

Example: `http://172.16.1.130:5000/`

### Server Files

| File | Purpose |
|------|---------|
| `api_server.py` | Main Flask application |
| `api_server.log` | Server logs |
| `api_server.pid` | Process ID file |
| `start.sh` | Start script |
| `stop.sh` | Stop script |
| `restart.sh` | Restart script |

---

## Pipeline Commands

```bash
cd /mnt/data/sri/wag/scripts

# Full pipeline
python run_pipeline.py --all

# Individual stages
python run_pipeline.py --prep      # Data preparation only
python run_pipeline.py --train     # Training only
python run_pipeline.py --eval      # Evaluation only
python run_pipeline.py --ollama    # Ollama setup

# Verification
python run_pipeline.py --check     # Check prerequisites
```

---

## Script Reference

### Data Preparation

| Script | Input | Output |
|--------|-------|--------|
| `extract_training_data.py` | WAG History.xlsx | wag_train/val/test.jsonl |
| `enrich_with_products.py` | Training JSON + Master Item | wag_enriched_*.jsonl |
| `validate_data.py` | Training JSON | validation_report.md |

```bash
cd /mnt/data/sri/wag/scripts/data_prep

# Extract training data
python extract_training_data.py \
    --input "../../WAG History.xlsx" \
    --output "../output/data"

# Enrich with product details
python enrich_with_products.py \
    --training "../output/data/wag_training_data_raw.json" \
    --master "../../WAG Master Item.xlsx" \
    --output "../output/data_enriched"

# Validate data quality
python validate_data.py \
    --input "../output/data/wag_training_data_raw.json" \
    --output "../output/reports/validation_report.md"
```

### Training

| Script | Input | Output |
|--------|-------|--------|
| `train.py` | Training JSONL + config.yaml | LoRA adapter |
| `evaluate.py` | Adapter + test data | Evaluation report |

```bash
cd /mnt/data/sri/wag/scripts/training

# Train model
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoint-500

# Evaluate model
python evaluate.py \
    --model ../output/models/wag-copywriter \
    --test ../output/data/wag_test.jsonl \
    --output ../output/reports/evaluation
```

### Inference

| Script | Input | Output |
|--------|-------|--------|
| `generate.py` | WIC codes, price, offer | Headline + body copy |
| `api_server.py` | HTTP requests | JSON responses |
| `ollama_setup.py` | Trained adapter | Ollama model |

```bash
cd /mnt/data/sri/wag/scripts/inference

# Single generation
python generate.py --wics "691500,691501" --price "$9.99" --offer "BOGO 50%"

# Batch from EAB file
python generate.py --eab "../../EABs/11.30 EAB.xls" --output results.json

# Using Ollama
python generate.py --wics "691500" --use-ollama
```

---

## Configuration Reference

### Training Config (`training/config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `model.name` | Mistral-7B-Instruct-v0.2 | Base model |
| `lora.r` | 64 | LoRA rank |
| `lora.lora_alpha` | 128 | LoRA scaling |
| `training.num_train_epochs` | 3 | Training epochs |
| `training.per_device_train_batch_size` | 4 | Batch size |
| `training.gradient_accumulation_steps` | 8 | Gradient accumulation |
| `training.learning_rate` | 2e-4 | Learning rate |
| `training.max_seq_length` | 512 | Max sequence length |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | http://localhost:11434 | Ollama API URL |
| `WAG_MODEL` | wag-copywriter | Default model |
| `HOST` | 0.0.0.0 | API server host |
| `PORT` | 5000 | API server port |

---

## Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | RTX 3080 (10GB) | RTX 4080 (16GB) | RTX 4090 (24GB) |
| GPU VRAM | 12 GB | 16 GB | 24 GB |
| System RAM | 32 GB | 64 GB | 128 GB |
| Storage | 50 GB SSD | 100 GB NVMe | 200 GB NVMe |

---

## Ollama Deployment

### Quick Setup (No Training)

```bash
cd /mnt/data/sri/wag/scripts/inference
python ollama_setup.py --setup-base
ollama run wag-copywriter-base
```

### After Fine-Tuning

```bash
python ollama_setup.py --create
python ollama_setup.py --test
ollama run wag-copywriter
```

### API Usage

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Price: $8.99 | Offer: BOGO 50%",
  "stream": false
}'
```

---

## Output Locations

```
scripts/output/
├── data/                          # Extracted training data
│   ├── wag_training_data_raw.json
│   ├── wag_train.jsonl            # 80% training
│   ├── wag_val.jsonl              # 10% validation
│   ├── wag_test.jsonl             # 10% test
│   └── extraction_stats.json
│
├── data_enriched/                 # Enriched with product details
│   ├── wag_enriched_full.json
│   ├── wag_enriched_train.jsonl
│   ├── wag_enriched_val.jsonl
│   └── wag_enriched_test.jsonl
│
├── models/wag-copywriter/         # Trained models
│   ├── adapter/                   # LoRA weights
│   ├── training_config.yaml
│   └── Modelfile
│
├── reports/                       # Reports
│   ├── validation_report.md
│   ├── validation_report.json
│   ├── evaluation.md
│   └── evaluation.json
│
├── checkpoints/                   # Training checkpoints
└── logs/                          # Training logs
```

---

## Troubleshooting

### API Server

```bash
# Port in use
sudo fuser -k 5000/tcp

# Check logs
tail -f /mnt/data/sri/wag/scripts/inference/api_server.log

# Network access issue
# Ensure --host 0.0.0.0 and check firewall
sudo ufw allow 5000
```

### Training

```yaml
# Out of Memory - reduce batch size in config.yaml:
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
```

### Ollama

```bash
# Start service
ollama serve

# Check models
ollama list

# Pull base model
ollama pull mistral:7b-instruct
```

### Dependencies

```bash
# Activate venv first!
source /mnt/data/sri/wag/scripts/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Excel support
pip install openpyxl xlrd
```

---

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 112,728 |
| Training Suitable | 97.5% |
| With Body Copy | 72.8% |
| Unique Products | 25,835 |
| Avg WICs per Example | 7.6 |

---

## Next Steps

1. **Check prerequisites:** `python run_pipeline.py --check`
2. **Start API server:** `./inference/start.sh`
3. **Test API:** `curl http://localhost:5000/api/health`
4. **Prepare data:** `python run_pipeline.py --prep`
5. **Train model:** `python run_pipeline.py --train`
6. **Deploy:** `python run_pipeline.py --ollama`

---

## Support

**Team:** Enterprise Architecture Team

**Documentation:**
- `scripts/README.md` - Detailed documentation
- `LLM_FINETUNING_PLAN.md` - Project planning
- `MODEL_GUIDE.md` - Model architecture & post-training roadmap
