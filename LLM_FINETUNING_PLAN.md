# WAG Retail Ad Copy LLM Fine-Tuning Plan

*Document Created: November 29, 2025*
*Last Updated: December 2025*
*Project: Walgreens Ad Headline & Body Copy Automation*

---

## Executive Summary

This document outlines the strategy for fine-tuning a local LLM to automate headline and body copy creation for Walgreens retail advertising campaigns. The analysis is based on historical campaign data, product master lists, and PDF ad samples available in the WAG directory.

**Goal:** Train a local LLM to generate marketing headlines and body copy based on product WIC codes, pricing, and promotion details.

---

## 1. Data Inventory

### 1.1 Available Data Assets

| Asset | File | Records | Size | Purpose |
|-------|------|---------|------|---------|
| Campaign History | `WAG History.xlsx` | 115,061 | 6.1 MB | **Primary training data** |
| Product Master | `WAG Master Item.xlsx` | 888,937 | 61.6 MB | Product context & embeddings |
| Campaign Templates | `EABs/*.xls` | ~130,000 | 99.6 MB | Template structure & pricing |
| Visual Samples | `PDFs/*` | 20 files | 213 MB | Validation & output reference |
| Database Backup | `WAG History.fmp12` | - | 47.3 MB | FileMaker archive |

**Total Data Volume:** 427.6 MB

### 1.2 WAG History Structure (Training Data)

Primary source for headline/body copy pairs.

| Column | Name | Description | ML Relevance |
|--------|------|-------------|--------------|
| 1 | EventName | Campaign event date/name | Context |
| 2 | StartDate | Campaign start date | Seasonality |
| 3 | EndDate | Campaign end date | Seasonality |
| 4 | ItemCodeGroup01 | Comma-separated WIC codes | **Input feature** |
| 5 | **Headline** | Marketing headline text | **Target output** |
| 6 | **BodyCopy** | Marketing body copy text | **Target output** |
| 7 | Disclaimer | Legal disclaimer text | Secondary target |
| 8 | AdRetail | Advertisement retail price | Input feature |
| 9 | SingleItemPrice | Single item price | Input feature |
| 10 | AdDollarOff | Dollar amount off | Promo context |
| 11 | PercentOff | Percent discount | Promo context |
| 12 | FinalPrice | Final sale price | Input feature |
| 13-14 | LoPrice/HighPrice | Price range | Input feature |
| 15 | Limit | Purchase limit | Body copy context |
| 16-30 | Various | Qty, BOGO, Rewards, FSI, IVC, MIR, SUR | Promo features |

### 1.3 WAG Master Item Structure (Product Context)

| Column | Name | Example | ML Use |
|--------|------|---------|--------|
| WIC | Walgreens Item Code | 691500 | Join key |
| UPC | Universal Product Code | 30085052202 | External lookup |
| Description | Product name | CORICIDIN HBP COLD/FLU TAB 24S | **Input feature** |
| CM | Category Manager | GEHRKE A | Grouping |
| Vendor | Product vendor | BAYER CONSUMER CARE | Brand context |
| Prod Cat Code | Category code | 001 | Classification |
| Brand | Brand name | Coricidin | **Input feature** |

### 1.4 EAB File Structure (Campaign Templates)

11 files with ~54 columns each, containing:

- Page/Layout positioning (Cols A-C)
- Product identifiers (WIC, UPC, Description)
- Regional pricing (US, DR, AK, HI, PR variants)
- Offer details (AO/LTY Type, FSI, Coupons, FARR)
- Cost information (Invoice, Departmental, Distribution)

### 1.5 Market Coverage

| Market Code | Region | Data Available |
|-------------|--------|----------------|
| NAT | National (Continental US) | Full coverage |
| AK | Alaska | Full coverage |
| HI | Hawaii | Full coverage |
| PR | Puerto Rico | Full coverage |

---

## 2. LLM Recommendation

### 2.1 Primary Recommendation: Mistral 7B Instruct

| Attribute | Value |
|-----------|-------|
| Model | Mistral 7B Instruct v0.2+ |
| Parameters | 7 billion |
| VRAM Required | 14-16 GB (QLoRA: 8-12 GB) |
| License | Apache 2.0 (commercial OK) |
| Fine-tuning Method | QLoRA (4-bit quantization) |

**Justification:**

1. **Dataset Size Match:** 115K training examples is optimal for 7B parameter models
2. **Task Complexity:** Headlines/body copy are short, structured text - doesn't need larger models
3. **Hardware Efficiency:** Runs on consumer GPUs (RTX 3080/4080/4090)
4. **Inference Speed:** Fast enough for production use
5. **Commercial License:** Apache 2.0 allows unrestricted commercial deployment
6. **Fine-tuning Support:** Excellent LoRA/QLoRA ecosystem (Unsloth, Axolotl, PEFT)

### 2.2 Alternative Models

| Model | Parameters | VRAM | Pros | Cons |
|-------|------------|------|------|------|
| **Llama 3.1 8B** | 8B | 16GB | Meta quality, strong reasoning | Slightly slower training |
| **Phi-3 Medium** | 14B | 28GB | Higher quality outputs | More hardware required |
| **Qwen2.5 7B** | 7B | 14GB | Excellent multilingual (Spanish for PR) | Less community tooling |
| **Gemma 2 9B** | 9B | 18GB | Concise, focused outputs | Google license restrictions |

### 2.3 Hardware Requirements

| Resource | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| GPU | RTX 3080 (10GB) | RTX 4080 (16GB) | RTX 4090 (24GB) |
| GPU VRAM | 12 GB | 16 GB | 24 GB |
| System RAM | 32 GB | 64 GB | 128 GB |
| Storage | 50 GB SSD | 100 GB NVMe | 200 GB NVMe |
| Training Time | 8-12 hours | 4-6 hours | 2-3 hours |

---

## 3. Fine-Tuning Strategy

### 3.1 Training Data Format

**Input Prompt Template:**
```
You are a retail advertising copywriter for Walgreens. Generate a headline and body copy for the following products and promotion.

Products:
- WIC: {wic_codes}
- Names: {product_descriptions}
- Brands: {brands}
- Category: {category}

Promotion Details:
- Price: {ad_retail}
- Offer: {offer_type} (e.g., BOGO 50%, $2 Off, Buy 2 Get 1 Free)
- Final Price: {final_price}
- Limit: {limit}

Generate a concise headline and body copy suitable for a print advertisement.
```

**Expected Output:**
```
Headline: {headline}
BodyCopy: {body_copy}
```

### 3.2 Sample Training Examples

**Example 1: OTC Medicine**
```json
{
  "input": "Products: CORICIDIN HBP COLD/FLU TAB 24S, EXCEDRIN TAB 100S | Brands: Coricidin, Excedrin | Price: $9.99 | Offer: BOGO 50%",
  "output": "Headline: Cold & Flu Relief\nBodyCopy: Select varieties. Limit 2."
}
```

**Example 2: Personal Care**
```json
{
  "input": "Products: SECRET CLINICAL IS FREE SENSITIVE 2.6OZ | Brand: Secret | Price: $12.99 | Offer: $3 Off",
  "output": "Headline: Secret Clinical Strength\nBodyCopy: Select varieties."
}
```

**Example 3: Multi-Product**
```json
{
  "input": "Products: BAUSCH + LOMB EYE CARE, BLINK EYE DROPS | Brands: Bausch + Lomb, Blink | Price: $7.99-$14.99 | Offer: Buy 1 Get 1 50%",
  "output": "Headline: Bausch + Lomb or Blink Eye Care\nBodyCopy: Select varieties."
}
```

### 3.3 Training Configuration

```yaml
# QLoRA Configuration (Recommended)
model_name: "mistralai/Mistral-7B-Instruct-v0.2"
quantization: "4bit"
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Parameters
batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2e-4
num_epochs: 3
max_seq_length: 512
warmup_ratio: 0.03

# Dataset Split
train_split: 0.80  # 92,048 records
val_split: 0.10    # 11,506 records
test_split: 0.10   # 11,506 records
```

### 3.4 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| BLEU Score | > 0.4 | N-gram overlap with reference |
| ROUGE-L | > 0.5 | Longest common subsequence |
| Exact Match | > 30% | Identical to historical copy |
| Human Approval | > 85% | Manual quality review |
| Character Count | Within 10% | Matches layout constraints |

---

## 4. Additional Use Cases

### 4.1 High-Value Use Cases

| # | Use Case | Description | Data Source | Priority |
|---|----------|-------------|-------------|----------|
| 1 | **Headline Generation** | Primary use case - generate ad headlines | WAG History | Critical |
| 2 | **Body Copy Generation** | Generate supporting body copy | WAG History | Critical |
| 3 | **Disclaimer Generation** | Auto-generate legal disclaimers | WAG History (Col 7) | High |
| 4 | **Regional Adaptation** | Adjust copy for AK/HI/PR markets | EAB regional data | High |
| 5 | **Promotion Description** | Explain offer types (BOGO, %, Rewards) | WAG History (Cols 10-30) | High |

### 4.2 Medium-Value Use Cases

| # | Use Case | Description | Benefit |
|---|----------|-------------|---------|
| 6 | **Product Categorization** | Auto-classify new products | Master Item (888K labels) |
| 7 | **A/B Headline Variants** | Generate multiple options per ad | Creative testing |
| 8 | **Character Optimization** | Fit copy to layout constraints | Layout automation |
| 9 | **Seasonal Adaptation** | Adjust tone for holidays | Campaign consistency |
| 10 | **Brand Voice Matching** | Learn vendor-specific language | Brand compliance |

### 4.3 Advanced Use Cases (Future)

| # | Use Case | Additional Data Needed | Complexity |
|---|----------|----------------------|------------|
| 11 | **Spanish Translation** | Parallel PR translations | Medium |
| 12 | **Layout Recommendation** | PDF parsing + position data | High |
| 13 | **Image-to-Copy** | Product images + OCR | High |
| 14 | **Performance Prediction** | Sales/engagement metrics | High |
| 15 | **Competitive Analysis** | Competitor ad data | Medium |

---

## 5. Implementation Roadmap

### Phase 1: Data Preparation (Week 1-2) ✅ COMPLETED

- [x] Extract WAG History to clean CSV/JSONL
- [x] Parse WIC codes from ItemCodeGroup01 column
- [x] Join with Master Item for product details
- [x] Create instruction-tuning dataset format
- [x] Implement train/val/test split (80/10/10)
- [x] Data quality validation and cleaning
- [x] Handle missing values and edge cases

**Deliverable:** `training_data.jsonl` (112,728 records - 97.5% training suitable)

### Phase 2: Environment Setup (Week 2) ✅ COMPLETED

- [x] Configure GPU environment (CUDA, cuDNN)
- [x] Install training framework (PyTorch + HuggingFace)
- [x] Download Mistral 7B Instruct base model
- [x] Set up experiment tracking
- [x] Configure Ollama for local inference
- [x] Test base model performance on sample prompts

**Deliverable:** Working training environment at `/mnt/data/sri/wag/scripts/venv/`

### Phase 3: Model Training (Week 3) 🔄 IN PROGRESS

- [ ] Initial training run with default hyperparameters
- [ ] Evaluate on validation set
- [ ] Hyperparameter tuning (learning rate, LoRA rank)
- [ ] Multiple training iterations
- [ ] Select best checkpoint based on metrics
- [ ] Export final LoRA adapter weights

**Deliverable:** Fine-tuned LoRA adapter

### Phase 4: Evaluation & Testing (Week 4)

- [ ] Quantitative evaluation (BLEU, ROUGE)
- [ ] Human evaluation on test set
- [ ] Edge case testing (new products, unusual promos)
- [ ] Regional variation testing (AK, HI, PR)
- [ ] Performance benchmarking (latency, throughput)
- [ ] Compare against PDF samples for accuracy

**Deliverable:** Evaluation report with metrics

### Phase 5: Integration & Deployment (Week 5) ✅ PARTIALLY COMPLETED

- [x] Deploy model via Ollama API
- [x] Build inference pipeline per Instructions.txt workflow
- [x] Build REST API server (Flask)
- [x] Create server management scripts (start/stop/restart)
- [ ] Integrate with EAB processing workflow
- [ ] Implement fallback for low-confidence outputs
- [ ] A/B testing against existing copy
- [x] Documentation and training for users

**Deliverable:** Production-ready API endpoint at `http://<server>:5000/`

### Phase 6: Monitoring & Iteration (Ongoing)

- [ ] Monitor output quality in production
- [ ] Collect feedback for retraining
- [ ] Periodic model updates with new data
- [ ] Expand to additional use cases

---

## 6. Technical Architecture

### 6.1 Training Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  WAG History    │────▶│  Data Parser    │────▶│  Training Data  │
│  (115K records) │     │  (Python/Pandas)│     │  (JSONL format) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│  Master Item    │────▶│  Context        │──────────────┘
│  (888K products)│     │  Enrichment     │
└─────────────────┘     └─────────────────┘

                        ┌─────────────────┐
                        │  Mistral 7B     │
                        │  Base Model     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  QLoRA Training │
                        │  (Unsloth)      │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Fine-tuned     │
                        │  LoRA Adapter   │
                        └─────────────────┘
```

### 6.2 Inference Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  EAB File       │────▶│  WIC Extractor  │────▶│  Product Lookup │
│  (New Campaign) │     │  (Page/Layout)  │     │  (Master Item)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  History Check  │◀─────────────┘
                        │  (Exact Match?) │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              [Match Found]             [No Match]
                    │                         │
                    ▼                         ▼
           ┌───────────────┐         ┌───────────────┐
           │ Return Cached │         │ LLM Generate  │
           │ Headline/Copy │         │ (Ollama API)  │
           └───────────────┘         └───────┬───────┘
                                             │
                                     ┌───────▼───────┐
                                     │ Human Review  │
                                     │ (If needed)   │
                                     └───────────────┘
```

### 6.3 Ollama Integration

```bash
# Model will be available at existing Ollama endpoint
# ollama.local:11434 (already configured in infrastructure)

# Load fine-tuned model
ollama create wag-copywriter -f Modelfile

# API call example
curl http://ollama.local:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Brand: Advil | Price: $8.99 | Offer: BOGO 50%",
  "stream": false
}'
```

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient training data quality | Medium | High | Data cleaning pipeline, manual review of samples |
| Model generates incorrect claims | Medium | High | Disclaimer templates, human review workflow |
| Hardware limitations | Low | Medium | Cloud GPU fallback (RunPod, Lambda Labs) |
| Poor regional adaptation | Medium | Medium | Separate fine-tuning for PR Spanish content |
| Model drift over time | Medium | Low | Periodic retraining with new campaigns |
| Latency in production | Low | Medium | Model quantization, caching common outputs |

---

## 8. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Headline accuracy | > 85% acceptable | Human review of 500 samples |
| Body copy accuracy | > 80% acceptable | Human review of 500 samples |
| Processing time | < 2 seconds/ad | API latency monitoring |
| Adoption rate | > 70% of new campaigns | Usage tracking |
| Time savings | > 50% reduction | Before/after comparison |
| Error rate | < 5% requiring major edits | Production monitoring |

---

## 9. Next Steps

1. **Immediate:** Run data preparation script to extract training pairs
2. **This Week:** Set up training environment and validate GPU access
3. **Next Week:** Begin initial fine-tuning experiments
4. **Week 3:** Evaluate results and iterate on training
5. **Week 4:** Integration testing with EAB workflow

---

## Appendix A: Sample Data Patterns

### Common Headline Patterns (from WAG History)

| Pattern | Example | Frequency |
|---------|---------|-----------|
| Brand Name Only | "Advil Pain Relief" | ~40% |
| Brand + Product Type | "Advil PM Pain Relief" | ~25% |
| Category Description | "Cold & Flu Relief" | ~15% |
| Multi-Brand | "Bausch + Lomb or Blink Eye Care" | ~10% |
| Promotional | "Holiday Gift Sets" | ~10% |

### Common Body Copy Patterns

| Pattern | Example | Frequency |
|---------|---------|-----------|
| Select varieties | "Select varieties." | ~60% |
| Select + Limit | "Select varieties. Limit 2." | ~20% |
| Empty/None | "" | ~15% |
| Specific description | "42 ct. capsules" | ~5% |

---

## Appendix B: File Locations

```
/mnt/data/sri/wag/
├── WAG History.xlsx              # Primary training data (115K records)
├── WAG Master Item.xlsx          # Product reference (888K products)
├── WAG History.fmp12             # FileMaker backup
├── Instructions.txt              # Processing workflow
├── LLM_FINETUNING_PLAN.md        # This document
├── SCRIPTS_OVERVIEW.md           # Quick reference guide
├── EABs/                         # Campaign templates (11 files)
│   ├── 1.4 EAB.xls
│   ├── 1.11 EAB.xls
│   ├── 1.18 EAB.xls
│   ├── 11.2 EAB.xls
│   ├── 11.9 EAB.xls
│   ├── 11.16 EAB.xls
│   ├── 11.23 EAB.xls
│   ├── 11.30 EAB.xls
│   ├── 1214 HI EAB.xls
│   ├── 1228 HI EAB.xls
│   └── 1221 PR EAB.xlsx
├── PDFs/                         # Campaign samples (20 PDFs)
│   ├── 11_02 All Markets/
│   ├── 11_09 All Markets/
│   ├── 11_16 All Markets/
│   ├── 11_23 All Markets/
│   └── 11_30 All Markets/
└── scripts/                      # Implementation code
    ├── README.md                 # Detailed documentation
    ├── requirements.txt          # Python dependencies
    ├── run_pipeline.py           # Master orchestration
    ├── venv/                     # Virtual environment
    ├── data_prep/                # Data preparation scripts
    ├── training/                 # Training scripts & config
    ├── inference/                # API server & generation
    │   ├── api_server.py         # Flask REST API
    │   ├── start.sh              # Start server
    │   ├── stop.sh               # Stop server
    │   └── restart.sh            # Restart server
    ├── utils/                    # Shared utilities
    └── output/                   # Generated outputs
        ├── data/                 # Training data
        ├── data_enriched/        # Enriched data
        ├── models/               # Trained models
        └── reports/              # Evaluation reports
```

---

## Appendix C: Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Base Model | Mistral 7B Instruct | v0.2+ |
| Training Framework | PyTorch + HuggingFace TRL | Latest |
| Quantization | BitsAndBytes | 0.41+ |
| PEFT Library | Hugging Face PEFT | 0.7+ |
| Inference Server | Ollama | 0.1.17+ |
| REST API | Flask + Flask-CORS | 3.0+ |
| Data Processing | Python/Pandas | 3.10+ |
| GPU Framework | CUDA | 12.1+ |

---

## Appendix D: API Server Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI with interactive testing |
| GET | `/api/health` | Health check and status |
| GET | `/api/models` | List available models |
| GET | `/api/training/status` | Fine-tuning progress |
| POST | `/api/generate` | Generate ad copy for products |
| POST | `/api/generate/batch` | Batch generation for multiple items |
| POST | `/api/generate/eab` | Process EAB slot data |

### Web UI Features

- **Model Selector** - Dropdown to choose from available Ollama models
- **Temperature Control** - Adjustable creativity setting (0-2)
- **Training Status Panel** - Real-time fine-tuning progress:
  - Progress bar with percentage
  - Steps, epochs, loss, elapsed time, ETA
  - Auto-refreshes every 5 seconds

### Server Management

```bash
cd /mnt/data/sri/wag/scripts/inference
./start.sh           # Start server (background)
./stop.sh            # Stop server
./restart.sh         # Restart server
```

### Network Access

Access from any machine on network: `http://<server-ip>:5000/`

---

*Document maintained by: Enterprise Architecture Team*
*Last Updated: December 2025*
