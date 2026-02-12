# tonsai

Khmer-English LLM training starter kit using **uv + Hugging Face + DeepSpeed**.

## What this repo provides
- GPT-2 style ~0.5B model config for continual pretraining or training from scratch.
- Multi-source Khmer/English dataset preparation with weighted mixing.
- Tokenizer training from prepared datasets or raw text files.
- Training script with DeepSpeed support, checkpoint controls, and optional Hub upload.
- Inference script for prompt completion.

## 1) Setup

```bash
uv sync
```

## 2) Data preparation

### 2.1 Build training mixture (YAML-driven)

```bash
uv run python scripts/prepare_dataset.py \
  --config configs/data/train_sources.example.yaml \
  --output-dir data/mixture_train
```

### 2.2 Build tokenizer mixture (smaller / cleaner)

```bash
uv run python scripts/prepare_dataset.py \
  --config configs/data/tokenizer_sources.example.yaml \
  --output-dir data/mixture_tokenizer
```

### 2.3 Legacy CLI mode

```bash
uv run python scripts/prepare_dataset.py \
  --khmer "path_or_hf_dataset_for_khmer" \
  --english "path_or_hf_dataset_for_english" \
  --khmer-weight 0.6 \
  --english-weight 0.4 \
  --output-dir data/mixture
```

## 3) Tokenizer training

### From prepared dataset

```bash
uv run python scripts/train_tokenizer.py \
  --dataset-dir data/mixture_tokenizer \
  --dataset-split train \
  --text-column text \
  --max-examples 1000000 \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

### From raw text files

```bash
uv run python scripts/train_tokenizer.py \
  --input data/khmer.txt data/english.txt \
  --vocab-size 50000 \
  --output-dir artifacts/tokenizer
```

## 4) Model training

### From scratch

```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture_train \
  --dataset-split train \
  --text-column text \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --warmup-steps 200 \
  --save-strategy steps \
  --save-steps 1000 \
  --save-total-limit 5 \
  --output-dir artifacts/checkpoints
```

### Continual pretraining

```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture_train \
  --dataset-split train \
  --text-column text \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --resume-from "existing_model_or_checkpoint" \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --warmup-steps 200 \
  --save-strategy steps \
  --save-steps 1000 \
  --save-total-limit 5 \
  --output-dir artifacts/checkpoints_cpt
```

## 5) Optional: Upload checkpoints/model to Hugging Face Hub

```bash
uv run huggingface-cli login
```

```bash
uv run python scripts/train_gpt2.py \
  --dataset-dir data/mixture_train \
  --dataset-split train \
  --text-column text \
  --tokenizer-dir artifacts/tokenizer \
  --model-config configs/model/gpt2_500m.json \
  --deepspeed configs/deepspeed/zero2_h100.json \
  --push-to-hub \
  --hub-model-id "your-username/khmer-gpt2-500m" \
  --hub-strategy every_save \
  --hub-private \
  --output-dir artifacts/checkpoints
```

## 6) Inference (prompt completion)

```bash
uv run python scripts/infer_gpt2.py \
  --model artifacts/checkpoints/final \
  --prompt "សួស្តី ខ្ញុំចង់សួរអំពី" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.95
```

Deterministic generation:

```bash
uv run python scripts/infer_gpt2.py \
  --model artifacts/checkpoints/final \
  --prompt "Write a short Khmer and English greeting:" \
  --no-sample
```

## 7) Key configs
- Model: `configs/model/gpt2_500m.json`
- DeepSpeed: `configs/deepspeed/zero2_h100.json`
- Training profile: `configs/train/cpt_500m.yaml`
- Data source templates:
  - `configs/data/train_sources.example.yaml`
  - `configs/data/tokenizer_sources.example.yaml`
