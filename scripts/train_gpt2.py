#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/continue pretraining GPT2 model")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--deepspeed", type=Path, default=Path("configs/deepspeed/zero2_h100.json"))
    parser.add_argument("--resume-from", type=str, default=None, help="Base model for continual pretraining")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, choices=["steps", "epoch"], default="steps")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    return parser.parse_args()


class OnTheFlyCausalCollator:
    def __init__(self, tokenizer, text_column: str, max_length: int):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __call__(self, features):
        texts = [f.get(self.text_column, "") for f in features]
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.model_max_length = args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(args.dataset_dir))[args.dataset_split]

    if args.resume_from:
        model = GPT2LMHeadModel.from_pretrained(args.resume_from)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model_config = GPT2Config.from_dict(json.loads(args.model_config.read_text()))
        model = GPT2LMHeadModel(model_config)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        deepspeed=str(args.deepspeed),
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=OnTheFlyCausalCollator(
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_length=args.max_length,
        ),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
