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

### 4.1 Why multi-source corpus design (llm-jp style)

Yes — your idea is correct. A multi-source design (Wikipedia + C4/mC4 + curated HQ corpora) is usually better than a single dataset because you get:
- stronger coverage,
- better factual diversity,
- easier future extension.

To support that, `scripts/prepare_dataset.py` now accepts a **YAML source config** with per-source weights and text column mapping.

### 4.2 Build mixed dataset from YAML (recommended)

Use the included extensible template:

```bash
uv run python scripts/prepare_dataset.py \
  --config configs/data/train_sources.example.yaml \
  --output-dir data/mixture_train
```

Template includes examples for:
- Khmer HQ seed: `kimleang123/khmer-text-dataset`
- Khmer web-scale: `allenai/c4` (`km` subset)
- Khmer Wikipedia-style source placeholder
- English HQ seed: `agentlans/high-quality-english-sentences`
- English web-scale: `allenai/c4` (`en` subset)

> Note: some dataset IDs/subsets may change or require replacement with current HF entries. The interface is designed so you only edit YAML.

### 4.3 Build tokenizer-only dataset (smaller, cleaner)

For tokenizer training, use less noisy but high-quality text first:

```bash
uv run python scripts/prepare_dataset.py \
  --config configs/data/tokenizer_sources.example.yaml \
  --output-dir data/mixture_tokenizer
```

### 4.4 Legacy CLI mode (still supported)

You can still pass Khmer/English sources directly:

```bash
uv run python scripts/prepare_dataset.py \
  --khmer "path_or_hf_dataset_for_khmer" \
  --english "path_or_hf_dataset_for_english" \
  --khmer-weight 0.6 \
  --english-weight 0.4 \
  --output-dir data/mixture
```

## 5) Train tokenizer

Use prepared dataset directly (recommended):

```bash
uv run python scripts/train_tokenizer.py \
  --dataset-dir data/mixture_tokenizer \
  --dataset-split train \
  --text-column text \
  --max-examples 1000000 \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

`--max-examples` is optional, but useful to speed up tokenizer iteration on very large corpora.

Or from raw text files:

```bash
uv run python scripts/train_tokenizer.py \
  --input data/khmer.txt data/english.txt \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

## 6) Pretraining / continual pretraining

### From scratch
```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture_train \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --output-dir artifacts/checkpoints
```

### Continual pretraining from existing model
```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture_train \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --resume-from "existing_model_or_checkpoint" \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --output-dir artifacts/checkpoints_cpt
```

## 7) Suggested evaluation plan
- Track train/validation perplexity separately for Khmer and English splits.
- Use a fixed Khmer prompt suite (QA, summarization, translation).
- Check script coverage and Unicode stability (Khmer normalization).
- Run toxicity/safety checks before deployment.

## 8) Practical notes for Colab + H100
- Prefer bf16 (already enabled in DeepSpeed config).
- Increase gradient accumulation if memory is tight.
- Save checkpoint every 500-1000 steps.
- Push intermediate checkpoints to cloud storage frequently.
- Start with pilot run first (few hundred million to 2B tokens) before full run.

## 9) Next steps
- Add data deduplication and quality scoring.
- Add packed-sequence training for throughput.
- Add instruction tuning recipes (LoRA/full SFT).
- Add Khmer-focused benchmark harness.
