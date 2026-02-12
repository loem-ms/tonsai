# tonsai

Khmer-English LLM training starter kit using **uv + Hugging Face + DeepSpeed**.

This repository is designed for your use case:
- GPT-2 style model around **0.5B parameters**.
- Mix Khmer and English corpora with controllable sampling.
- Run on H100 (including Colab environments where available).
- Support both:
  1) training from scratch, and
  2) continual pretraining from an existing checkpoint.

## 1) Recommended strategy

### Stage A: Data pipeline first (fast iteration)
1. Start with available dumps (Khmer + English) and normalize text.
2. Train tokenizer on mixed corpus (Khmer-heavy at first, e.g. 60/40).
3. Build a small pilot dataset and run 0.5-2B token pilot training.
4. Evaluate Khmer perplexity + benchmark prompts before scaling.

### Stage B: Main pretraining
- Target ~100B+ tokens if possible for stable 0.5B model behavior.
- Keep Khmer over-sampled relative to natural web frequency.
- Suggested data ratios for first production run:
  - **Khmer 40-60%**
  - **English 40-60%**

### Stage C: SFT / instruction tuning
After base model pretraining, run supervised fine-tuning on Khmer/English instruction data for better chat behavior.

## 2) Architecture recommendation (GPT-2 ~0.5B)

`configs/model/gpt2_500m.json` defines a GPT-2-like decoder model:
- 24 layers
- hidden size 1536
- 24 heads
- context length 2048
- vocab size 50k

This is in the ~0.5B parameter class and works as a practical first target.

## 3) Environment setup with uv

```bash
uv sync
```

For Colab terminal/Jupyter usage:
1. Clone this repo.
2. Install uv.
3. Run `uv sync`.
4. Launch training with `uv run ...` commands below.

## 4) Data preparation

### 4.1 Build mixed dataset

```bash
uv run python scripts/prepare_dataset.py \
  --khmer "path_or_hf_dataset_for_khmer" \
  --english "path_or_hf_dataset_for_english" \
  --khmer-weight 0.6 \
  --english-weight 0.4 \
  --output-dir data/mixture
```

### 4.2 Train tokenizer

Use the mixed dataset directly (recommended, no extra `khmer.txt` / `english.txt` files needed):

```bash
uv run python scripts/train_tokenizer.py \
  --dataset-dir data/mixture \
  --dataset-split train \
  --text-column text \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

If you already have raw text files, you can still train with:

```bash
uv run python scripts/train_tokenizer.py \
  --input data/khmer.txt data/english.txt \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

## 5) Pretraining / continual pretraining

### From scratch
```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --output-dir artifacts/checkpoints
```

### Continual pretraining from existing model
```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --resume-from "existing_model_or_checkpoint" \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --output-dir artifacts/checkpoints_cpt
```

## 6) Suggested evaluation plan
- Track train/validation perplexity separately for Khmer and English splits.
- Use a fixed Khmer prompt suite (QA, summarization, translation).
- Check script coverage and Unicode stability (Khmer normalization).
- Run toxicity/safety checks before deployment.

## 7) Practical notes for Colab + H100
- Prefer bf16 (already enabled in DeepSpeed config).
- Increase gradient accumulation if memory is tight.
- Save checkpoint every 500-1000 steps.
- Push intermediate checkpoints to cloud storage frequently.

## 8) Next steps
- Add data deduplication and quality scoring.
- Add packed-sequence training for throughput.
- Add instruction tuning recipes (LoRA/full SFT).
- Add Khmer-focused benchmark harness.
