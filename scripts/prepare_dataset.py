#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset


SourceConfig = dict[str, Any]


def _log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [prepare_dataset] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Khmer-English mixture dataset")
    parser.add_argument("--khmer", nargs="+", default=None, help="Khmer dataset spec(s): local text path or HF dataset id")
    parser.add_argument("--english", nargs="+", default=None, help="English dataset spec(s): local text path or HF dataset id")
    parser.add_argument("--khmer-weight", type=float, default=0.5, help="Sampling weight for Khmer")
    parser.add_argument("--english-weight", type=float, default=0.5, help="Sampling weight for English")
    parser.add_argument("--config", type=Path, default=None, help="YAML config with extensible dataset source list")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mixture"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.config and not args.khmer and not args.english:
        parser.error("Provide --config or at least one dataset via --khmer/--english.")
    return args


def _load_dataset_source(source: SourceConfig) -> Dataset:
    source_name = source.get("name") or source.get("dataset") or source.get("local_text")
    source_lang = source.get("language", "unknown")
    _log(f"Loading source: {source_lang}:{source_name}")

    if "local_text" in source:
        loaded = load_dataset("text", data_files={"train": source["local_text"]})["train"]
    else:
        dataset_name = source["dataset"]
        subset = source.get("subset")
        split = source.get("split", "train")
        if subset is not None:
            loaded = load_dataset(dataset_name, subset, split=split)
        else:
            loaded = load_dataset(dataset_name, split=split)

    text_column = source.get("text_column", "text")
    if text_column != "text":
        loaded = loaded.rename_column(text_column, "text")

    loaded = loaded.select_columns(["text"])
    _log(f"Loaded rows for {source_lang}:{source_name}: {len(loaded)}")
    return loaded


def _load_yaml_sources(config_path: Path) -> tuple[list[Dataset], list[float], list[str]]:
    _log(f"Reading source config: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    sources: list[SourceConfig] = config.get("sources", [])
    if not sources:
        raise ValueError("Config must contain a non-empty 'sources' list.")

    datasets_list: list[Dataset] = []
    weights: list[float] = []
    names: list[str] = []

    for source in sources:
        ds = _load_dataset_source(source)
        language = source.get("language", "unknown")
        if language not in {"khmer", "english"}:
            raise ValueError(f"Unsupported language '{language}' in config: {source}")

        datasets_list.append(ds)
        weights.append(float(source.get("weight", 1.0)))
        source_name = source.get("name") or source.get("dataset") or source.get("local_text")
        names.append(f"{language}:{source_name}")

    return datasets_list, weights, names


def _load_any(spec: str, language: str) -> Dataset:
    _log(f"Loading CLI source: {language}:{spec}")
    if Path(spec).exists():
        ds = load_dataset("text", data_files={"train": spec})["train"]
    else:
        ds = load_dataset(spec, split="train")
    _log(f"Loaded rows for {language}:{spec}: {len(ds)}")
    return ds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Output directory: {args.output_dir}")
    _log(f"Seed: {args.seed}")

    if args.config:
        datasets_list, weights, names = _load_yaml_sources(args.config)
    else:
        kh_specs = args.khmer or []
        en_specs = args.english or []

        kh = concatenate_datasets([_load_any(spec, "khmer") for spec in kh_specs]) if kh_specs else None
        en = concatenate_datasets([_load_any(spec, "english") for spec in en_specs]) if en_specs else None

        datasets_list = [d for d in [kh, en] if d is not None]
        if not datasets_list:
            raise ValueError("No datasets were provided from --khmer/--english.")
        weights = [args.khmer_weight] * (1 if kh is not None else 0) + [args.english_weight] * (1 if en is not None else 0)
        names = ["khmer:cli"] * (1 if kh is not None else 0) + ["english:cli"] * (1 if en is not None else 0)

    _log(f"Interleaving {len(datasets_list)} dataset(s) with weights={weights}")
    mixed = interleave_datasets(
        datasets_list,
        probabilities=weights,
        seed=args.seed,
        stopping_strategy="all_exhausted",
    )

    metadata = {
        "seed": args.seed,
        "sources": names,
        "weights": weights,
        "config_path": str(args.config) if args.config else None,
        "num_rows_train": len(mixed),
    }

    _log("Saving dataset to disk...")
    DatasetDict({"train": mixed}).save_to_disk(str(args.output_dir))
    (args.output_dir / "mixture_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    _log(f"Done. Saved train rows: {len(mixed)}")


if __name__ == "__main__":
    main()
