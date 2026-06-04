#!/usr/bin/env python
# coding=utf-8
"""
Data preprocessing for MLM fine-tuning on the ActDisease dataset.

Loads raw CSV data, chunks long texts, splits into train/validation,
and writes line-by-line text files for use with run_mlm.py.

Usage:
    python prepare_data.py [configs/prepare_data.yaml]
"""

import sys
import re
import yaml
import pandas as pd
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/prepare_data.yaml"
    cfg = load_config(config_path)

    # Load dataset
    dataset = pd.read_csv(cfg["input_file"], index_col=0)
    dataset.columns = ["text", "journal"]

    # Tokenizer used only to measure sequence length
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
    max_length = tokenizer.model_max_length

    def span_len(text: str) -> int:
        return len(tokenizer.encode(text))

    dataset["ntokens"] = dataset.text.apply(span_len)

    # Split documents that exceed 2*(max_length-2) tokens into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        length_function=span_len,
        separators=cfg["separators"],
    )

    split_texts = []

    def chunk_text(row):
        if row.ntokens >= (max_length - 2) * 2:
            for chunk in text_splitter.split_text(row.text):
                split_texts.append((chunk, row.journal))
        else:
            split_texts.append((row.text, row.journal))

    dataset.apply(chunk_text, axis=1)
    dataset = pd.DataFrame(split_texts, columns=["text", "journal"])

    # Strip leading non-word characters
    dataset["text"] = dataset.text.apply(lambda t: re.sub(r"^[\W\s_]*", "", t))

    # Stratified train/validation split
    train, valid = train_test_split(
        dataset,
        test_size=cfg["test_size"],
        stratify=dataset["journal"],
        random_state=cfg["random_state"],
    )

    print("Train journal distribution:")
    print(train.journal.value_counts())
    print(f"\nTrain samples : {len(train)}")
    print(f"Valid samples : {len(valid)}")

    # Write line-by-line text files
    for path, df in [(cfg["train_output"], train), (cfg["val_output"], valid)]:
        with open(path, "w") as f:
            for text in df.text.tolist():
                f.write(text + "\n")
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
