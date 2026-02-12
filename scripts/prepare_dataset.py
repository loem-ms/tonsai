#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Khmer-English mixture dataset")
    parser.add_argument("--khmer", nargs="+", required=True, help="Khmer dataset spec(s): path or hub id")
    parser.add_argument("--english", nargs="+", required=True, help="English dataset spec(s): path or hub id")
    parser.add_argument("--khmer-weight", type=float, default=0.5, help="Sampling weight for Khmer")
    parser.add_argument("--english-weight", type=float, default=0.5, help="Sampling weight for English")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mixture"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_any(spec: str) -> Dataset:
    if Path(spec).exists():
        return load_dataset("text", data_files={"train": spec})["train"]
    return load_dataset(spec)["train"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kh = concatenate_datasets([_load_any(spec) for spec in args.khmer])
    en = concatenate_datasets([_load_any(spec) for spec in args.english])

    mixed = interleave_datasets(
        [kh, en],
        probabilities=[args.khmer_weight, args.english_weight],
        seed=args.seed,
        stopping_strategy="all_exhausted",
    )

    DatasetDict({"train": mixed}).save_to_disk(str(args.output_dir))


if __name__ == "__main__":
    main()
