#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Khmer-English byte-level BPE tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input text files")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    return parser.parse_args()


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

    tokenizer.train(files=args.input, trainer=trainer)
    tokenizer.save(str(args.output_dir / "tokenizer.json"))
    tokenizer.model.save(str(args.output_dir))


if __name__ == "__main__":
    main()
