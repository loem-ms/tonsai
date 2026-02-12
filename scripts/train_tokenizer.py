#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Khmer-English byte-level BPE tokenizer")
    parser.add_argument("--input", nargs="+", default=None, help="Input text files")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Hugging Face dataset directory created by scripts/prepare_dataset.py",
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    if not args.input and not args.dataset_dir:
        parser.error("Provide either --input files or --dataset-dir.")
    return args


def _iter_dataset_text(dataset_dir: Path, split: str, text_column: str):
    dataset = load_from_disk(str(dataset_dir))[split]
    for row in dataset:
        text = row.get(text_column)
        if text:
            yield text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("\u200b", ""),
        normalizers.Replace("\ufeff", ""),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    if args.dataset_dir:
        iterator = _iter_dataset_text(args.dataset_dir, args.dataset_split, args.text_column)
        tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
    else:
        tokenizer.train(files=args.input, trainer=trainer)

    tokenizer.save(str(args.output_dir / "tokenizer.json"))
    tokenizer.model.save(str(args.output_dir))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.output_dir / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    hf_tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
