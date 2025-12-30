# WAG-Copywriter Model Guide

*Last Updated: December 2025*

This document explains the WAG-Copywriter model architecture, how it differs from the base Mistral model, and the roadmap for deployment after fine-tuning.

---

## Table of Contents

- [Model Overview](#model-overview)
- [Base Model vs WAG-Copywriter](#base-model-vs-wag-copywriter)
- [Customizations](#customizations)
- [Fine-Tuning Configuration](#fine-tuning-configuration)
- [Post-Training Roadmap](#post-training-roadmap)
- [Deployment Options](#deployment-options)
- [Model Versions](#model-versions)

---

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Mistral 7B Instruct v0.2 |
| Parameters | 7 billion |
| Fine-Tuning Method | QLoRA (4-bit quantization) |
| Training Data | 112,728 Walgreens campaign records |
| Purpose | Retail ad headline & body copy generation |

---

## Base Model vs WAG-Copywriter

| Aspect | Base Mistral 7B | WAG-Copywriter |
|--------|-----------------|----------------|
| **Model Type** | General-purpose LLM | Specialized for retail ad copy |
| **Training Data** | General web data | 112K Walgreens campaigns |
| **Output Style** | Free-form responses | Structured Headline/BodyCopy format |
| **Domain Knowledge** | None specific | Walgreens brands, products, promotions |
| **Adapter** | None | LoRA adapter (~100MB) |

### What the Fine-Tuning Teaches

The model learns from real Walgreens advertising examples:

1. **Brand Patterns**
   - Single brand: "Advil Pain Relief"
   - Multi-brand: "Bausch + Lomb or Blink Eye Care"
   - Category: "Cold & Flu Relief"

2. **Body Copy Conventions**
   - Standard: "Select varieties."
   - With limits: "Select varieties. Limit 2."
   - Specific: "42 ct. capsules"

3. **Promotional Language**
   - BOGO offers
   - Percentage discounts
   - Walgreens Rewards messaging

---

## Customizations

### 1. System Prompt

The model is configured with a specialized system prompt:

```
You are a retail advertising copywriter for Walgreens. Generate concise,
effective headlines and body copy for print advertisements.

Based on the products, pricing, and promotions provided, create:
1. A clear, brand-focused headline (typically the brand or product category name)
2. Brief body copy (usually "Select varieties." with any purchase limits)
```

### 2. Output Format

Trained to produce consistent, parseable output:

```
Headline: [Your headline here]
BodyCopy: [Your body copy here]
```

### 3. Generation Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.7 | Balanced creativity/consistency |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Vocabulary constraint |
| `repeat_penalty` | 1.1 | Reduce repetition |
| `num_ctx` | 2048 | Context window size |

---

## Fine-Tuning Configuration

### LoRA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` (rank) | 64 | Model capacity for new knowledge |
| `lora_alpha` | 128 | Scaling factor (typically 2x rank) |
| `lora_dropout` | 0.05 | Regularization |
| `bias` | none | No bias training |

### Target Modules

All attention and MLP layers are fine-tuned:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP)

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 4 per device |
| Gradient Accumulation | 8 steps |
| Effective Batch Size | 32 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 3% |
| Max Sequence Length | 512 tokens |
| Quantization | 4-bit NF4 (QLoRA) |

### Training Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 112,728 |
| Training Split | 80% (~90,182) |
| Validation Split | 10% (~11,273) |
| Test Split | 10% (~11,273) |
| Unique Products | 25,835 |

---

## Post-Training Roadmap

### Phase 1: Evaluation (After Training Completes)

```bash
cd /mnt/data/sri/wag/scripts/training

# Run evaluation on test set
python evaluate.py \
    --model ../output/models/wag-copywriter \
    --test ../output/data/wag_test.jsonl \
    --output ../output/reports/evaluation
```

**Expected Outputs:**
- `evaluation.md` - Human-readable report
- `evaluation.json` - Detailed metrics (BLEU, ROUGE, exact match)

**Target Metrics:**
| Metric | Target |
|--------|--------|
| BLEU Score | > 0.4 |
| ROUGE-L | > 0.5 |
| Exact Match | > 30% |

### Phase 2: Model Export

#### Option A: Use with Ollama (Recommended)

```bash
cd /mnt/data/sri/wag/scripts/inference

# Create Ollama model from fine-tuned adapter
python ollama_setup.py --create

# Verify model is available
ollama list

# Test the model
python ollama_setup.py --test
```

#### Option B: Convert to GGUF (For Standalone Use)

```bash
# Install llama.cpp tools
pip install llama-cpp-python

# Convert adapter to GGUF format
# (Instructions vary based on llama.cpp version)
```

### Phase 3: Testing

```bash
# Test via CLI
cd /mnt/data/sri/wag/scripts/inference
python generate.py --wics "691500,691501" --price "$9.99" --offer "BOGO 50%"

# Test via API
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "products": [{"description": "ADVIL PM 20CT", "brand": "Advil"}],
    "price": "$8.99",
    "offer": "BOGO 50%",
    "model": "wag-copywriter"
  }'
```

### Phase 4: Validation

1. **Automated Testing**
   - Run on 100+ sample inputs
   - Check output format consistency
   - Measure generation latency

2. **Human Review**
   - Review 50 random outputs
   - Check brand accuracy
   - Verify body copy appropriateness

3. **A/B Comparison**
   - Compare with base model outputs
   - Compare with historical human-written copy

### Phase 5: Production Deployment

1. **Update API Server**
   ```bash
   # Restart API server to use new model
   cd /mnt/data/sri/wag/scripts/inference
   ./restart.sh
   ```

2. **Set as Default Model**
   ```bash
   export WAG_MODEL=wag-copywriter
   ```

3. **Monitor Performance**
   - Check API response times
   - Monitor generation quality
   - Track error rates

---

## Deployment Options

### Option 1: Ollama API (Recommended)

**Pros:** Easy setup, good performance, REST API
**Cons:** Requires Ollama installation

```bash
# After training, create model
python ollama_setup.py --create

# Use via API
curl http://localhost:11434/api/generate -d '{
  "model": "wag-copywriter",
  "prompt": "Products: ADVIL PM 20CT | Price: $8.99",
  "stream": false
}'
```

### Option 2: Flask API Server

**Pros:** Custom endpoints, web UI, training status
**Cons:** Depends on Ollama backend

```bash
# Start server
./start.sh

# Access at
http://<server-ip>:5000/
```

### Option 3: Direct Python Inference

**Pros:** No external dependencies
**Cons:** More complex, requires GPU

```bash
python generate.py --model ../output/models/wag-copywriter --wics "691500"
```

---

## Model Versions

### wag-copywriter-base

- **Type:** Base Mistral with custom system prompt (no fine-tuning)
- **Use Case:** Immediate use, testing
- **Quality:** Good but not optimized for Walgreens style

```bash
ollama create wag-copywriter-base -f Modelfile.base
```

### wag-copywriter (After Fine-Tuning)

- **Type:** Fine-tuned with LoRA adapter
- **Use Case:** Production ad copy generation
- **Quality:** Optimized for Walgreens conventions

```bash
python ollama_setup.py --create
```

---

## Checklist: After Training Completes

- [ ] Check training status shows "completed" in web UI
- [ ] Review final loss value (should be < 1.0)
- [ ] Run evaluation script
- [ ] Review evaluation metrics
- [ ] Create Ollama model from adapter
- [ ] Test with sample inputs
- [ ] Human review of 50 outputs
- [ ] Update API server to use new model
- [ ] Document model version and metrics
- [ ] Archive training logs and checkpoints

---

## Files Reference

| File | Purpose |
|------|---------|
| `training/config.yaml` | Training configuration |
| `inference/Modelfile` | Ollama model definition (fine-tuned) |
| `inference/Modelfile.base` | Ollama model definition (base) |
| `output/models/wag-copywriter/adapter/` | LoRA adapter weights |
| `output/training_status.json` | Training progress |
| `output/reports/evaluation.md` | Evaluation results |

---

## Support

**Team:** Enterprise Architecture Team
**Related Docs:**
- `scripts/README.md` - Full documentation
- `LLM_FINETUNING_PLAN.md` - Project planning
- `SCRIPTS_OVERVIEW.md` - Quick reference
