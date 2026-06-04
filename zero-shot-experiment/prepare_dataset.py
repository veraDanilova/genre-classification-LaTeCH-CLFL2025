#!/usr/bin/env python
# coding=utf-8
"""
Prepare train/test splits from CORE, FTD, UDM, or all three (merged).

Usage:
    python prepare_dataset.py configs/datasets/CORE.yaml
    python prepare_dataset.py configs/datasets/merged.yaml

Each config produces:
    <output_dir>/<config_id>/train.csv
    <output_dir>/<config_id>/test.csv

Config ID is derived from the dataset name and key settings so that
different configurations do not overwrite each other.
"""

import os
import sys
import logging
import yaml
import pandas as pd

from dataset_utils import generate_dataset, download_CORE, download_FTD

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_id(cfg: dict) -> str:
    """Derive a unique folder name from the config settings."""
    ds   = cfg["dataset"]
    lff  = "lff" if cfg.get("language_family_filter") else "all"
    map_ = cfg.get("mapping", "cl_map_v3").replace("cl_map_", "v")
    ch   = "chunked" if cfg.get("chunking", True) else "raw"
    bal  = "balanced" if cfg.get("balance", True) else "imbalanced"
    return f"{ds}__{lff}__{map_}__{ch}__{bal}"


def process_single(ds: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline for one dataset."""
    logger.info(f"Processing {ds} …")
    return generate_dataset(
        ds=ds,
        data_dir=cfg["data_dir"],
        mapping=cfg.get("mapping", "cl_map_v3"),
        language_family_filter=cfg.get("language_family_filter") or None,
        chunking=cfg.get("chunking", True),
        model_name=cfg.get("model_name", "FacebookAI/xlm-roberta-base"),
        remove_punctuation=cfg.get("remove_punctuation", False),
        remove_short=cfg.get("remove_short", True),
        balance=cfg.get("balance", True),
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/datasets/CORE.yaml"
    cfg = load_config(config_path)

    # Optionally download raw data
    if cfg.get("download", False):
        logger.info("Downloading CORE …")
        download_CORE(cfg["data_dir"])
        logger.info("Downloading FTD …")
        download_FTD(cfg["data_dir"])

    dataset = cfg["dataset"]
    out_root = cfg.get("output_dir", "data/prepared")

    if dataset == "merged":
        # Process each source independently, then concatenate before splitting
        from dataset_utils import (
            load_dataset, map_labels, basic_preprocessing,
            chunk_and_filter, balance_dataset, split_train_test,
        )
        import pandas as pd

        frames = []
        for ds in ("CORE", "FTD", "UDM"):
            try:
                df = load_dataset(ds, cfg["data_dir"])
                df = map_labels(df, ds, cfg.get("mapping", "cl_map_v3"))
                df = basic_preprocessing(df)
                if cfg.get("chunking", True):
                    df = chunk_and_filter(
                        df,
                        model_name=cfg.get("model_name", "FacebookAI/xlm-roberta-base"),
                        language_family_filter=cfg.get("language_family_filter") or None,
                        remove_punctuation=cfg.get("remove_punctuation", False),
                        remove_short=cfg.get("remove_short", True),
                    )
                elif cfg.get("language_family_filter"):
                    lff = cfg["language_family_filter"]
                    df = df.query("language_family in @lff").copy()
                frames.append(df)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {ds}: {e}")

        merged = pd.concat(frames, ignore_index=True)
        if cfg.get("balance", True):
            merged = balance_dataset(merged, random_state=cfg.get("random_state", 42))
        train, test = split_train_test(merged,
                                       test_size=cfg.get("test_size", 0.2),
                                       random_state=cfg.get("random_state", 42))
    else:
        train, test = process_single(dataset, cfg)

    cid = config_id(cfg)
    out_dir = os.path.join(out_root, cid)
    os.makedirs(out_dir, exist_ok=True)

    train.to_csv(os.path.join(out_dir, "train.csv"))
    test.to_csv(os.path.join(out_dir, "test.csv"))

    logger.info(f"Saved to {out_dir}  (train={len(train)}, test={len(test)})")
    logger.info("Label distribution (train):\n" + train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
