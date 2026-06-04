#!/usr/bin/env python
# coding=utf-8
"""
Prepare few-shot training splits from an annotated ActDisease CSV.

Input CSV must have columns: language, label, text
(identical structure to data_few-shot-data/all__.train.csv)

Produces in output_dir/:
    few_shot.dev.csv         – held-out test/dev set (kept fixed across all sizes)
    100__.train.csv
    200__.train.csv
    300__.train.csv
    400__.train.csv
    500__.train.csv
    all__.train.csv          – full training pool (complement of dev set)

Usage:
    python prepare_data.py [configs/prepare_data.yaml]
"""

import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stratified_sample(df: pd.DataFrame, n: int, label_col: str, random_state: int) -> pd.DataFrame:
    """Sample exactly n rows with approximate label stratification.
    Falls back to plain sampling if any class is too small."""
    try:
        return df.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, round(n * len(g) / len(df))),
                replace=False,
                random_state=random_state,
            )
        ).sample(frac=1, random_state=random_state).head(n).reset_index(drop=True)
    except ValueError:
        return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/prepare_data.yaml"
    cfg = load_config(config_path)

    input_file   = cfg["input_file"]
    test_file    = cfg.get("test_file")      # optional: pre-split test set
    output_dir   = cfg.get("output_dir", "data_few-shot-data")
    sizes        = cfg.get("sizes", [100, 200, 300, 400, 500])
    test_size    = cfg.get("test_size", 0.3)
    random_state = cfg.get("random_state", 42)
    label_col    = cfg.get("label_col", "label")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file, index_col=0)
    required = {"language", label_col, "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}. Got: {list(df.columns)}")

    df = df.dropna(subset=["text", label_col]).drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} samples after deduplication.")
    print("Label distribution:\n" + df[label_col].value_counts().to_string())

    # Split off dev/test set
    if test_file:
        dev = pd.read_csv(test_file, index_col=0)
        train_pool = df
        print(f"\nUsing existing test file: {test_file}  ({len(dev)} samples)")
    else:
        train_pool, dev = train_test_split(
            df,
            test_size=test_size,
            stratify=df[label_col],
            random_state=random_state,
        )
        print(f"\nSplit: {len(train_pool)} train pool / {len(dev)} dev")

    dev_path = os.path.join(output_dir, "few_shot.dev.csv")
    dev.to_csv(dev_path)
    print(f"Dev set saved → {dev_path}")

    # Full training pool
    all_path = os.path.join(output_dir, "all__.train.csv")
    train_pool.to_csv(all_path)
    print(f"Full train saved → {all_path}  ({len(train_pool)} samples)")

    # N-shot subsets
    for n in sizes:
        if n > len(train_pool):
            print(f"  Skipping size {n}: train pool too small ({len(train_pool)})")
            continue
        sample = stratified_sample(train_pool, n, label_col, random_state)
        out_path = os.path.join(output_dir, f"{n}__.train.csv")
        sample.to_csv(out_path)
        print(f"  {n}-shot saved → {out_path}  (label dist: {dict(sample[label_col].value_counts())})")


if __name__ == "__main__":
    main()
