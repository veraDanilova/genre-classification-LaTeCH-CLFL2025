#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune a sequence classifier for zero-shot genre prediction.

Usage:
    python train.py configs/training/xlmroberta.yaml

GPU selection:
    CUDA_VISIBLE_DEVICES=0 python train.py configs/training/xlmroberta.yaml

The script expects a train.csv produced by prepare_dataset.py.
It performs a stratified 90/10 train/validation split internally and
evaluates on the held-out test.csv if --test_file is set in the config.
"""

import os
import sys
import logging
import yaml

import torch
import datasets as ds
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
import sklearn.metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def chunk_dataframe(df: pd.DataFrame, tokenizer, chunk_size: int = 490, chunk_overlap: int = 20) -> pd.DataFrame:
    """Chunk texts longer than 512 tokens (applied to already-tokenized data)."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def span_len(text):
        return len(tokenizer.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=span_len,
        separators=["\n\n", "\n", ". ", ";", ",", ":"],
    )
    df = df.copy()
    df["ntokens"] = df["text"].apply(span_len)

    rows = []
    for _, row in df.iterrows():
        if row["ntokens"] > 510:
            for chunk in splitter.split_text(row["text"]):
                rows.append({**row.to_dict(), "text": chunk})
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def build_hf_dataset(df: pd.DataFrame, tokenizer, labels: list, label2id: dict) -> ds.Dataset:
    """Tokenise a DataFrame and return a HuggingFace Dataset."""
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].map(label2id)
    dataset = ds.Dataset.from_pandas(df)
    label_feature = ds.ClassLabel(num_classes=len(labels), names=labels)
    dataset = dataset.cast_column("label", label_feature)
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def save_confusion_matrix(true_labels, pred_labels, id2label: dict, output_dir: str):
    class_labels = [id2label[i] for i in sorted(id2label)]
    cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    plt.figure(figsize=(10, 8))
    g = sns.heatmap(cm_df, cmap="hot_r", annot=True, fmt="g")
    g.xaxis.set_ticks_position("top")
    g.tick_params(axis="x", rotation=90)
    g.set_xlabel("True Label")
    g.set_ylabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/xlmroberta.yaml"
    cfg = load_config(config_path)

    train_path = cfg["train_file"]
    model_name = cfg.get("model_name", "FacebookAI/xlm-roberta-base")
    output_dir = cfg.get("output_dir", "output")

    run_name = os.path.splitext(os.path.basename(train_path))[0]
    run_name += f"__{model_name.split('/')[-1]}"
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading training data from {train_path} …")
    train_df = pd.read_csv(train_path, index_col=0)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if cfg.get("chunk_at_training", False):
        logger.info("Chunking long texts …")
        train_df = chunk_dataframe(train_df, tokenizer)

    labels = sorted(train_df["label"].dropna().unique().tolist())
    id2label = dict(enumerate(labels))
    label2id = {l: i for i, l in id2label.items()}

    full_dataset = build_hf_dataset(train_df, tokenizer, labels, label2id)
    split = full_dataset.train_test_split(
        test_size=cfg.get("valid_split", 0.1),
        stratify_by_column="label",
        seed=cfg.get("random_state", 42),
    )
    train_dataset, valid_dataset = split["train"], split["test"]

    logger.info(f"Train: {len(train_dataset)}  Valid: {len(valid_dataset)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    model.to(device)

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.get("learning_rate", 1e-5),
        per_device_train_batch_size=cfg.get("batch_size", 8),
        per_device_eval_batch_size=cfg.get("batch_size", 8),
        num_train_epochs=cfg.get("num_epochs", 5),
        weight_decay=cfg.get("weight_decay", 0.01),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
    )

    logger.info("Training …")
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "best_model"))
    logger.info(f"Model saved to {out_dir}/best_model")

    # Evaluate on internal validation set
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Evaluate on held-out test set (if provided)
    test_path = cfg.get("test_file")
    if test_path and os.path.exists(test_path):
        logger.info(f"Evaluating on test set: {test_path}")
        test_df = pd.read_csv(test_path, index_col=0)
        if cfg.get("chunk_at_training", False):
            test_df = chunk_dataframe(test_df, tokenizer)
        test_dataset = build_hf_dataset(test_df, tokenizer, labels, label2id)
        predictions = trainer.predict(test_dataset)
        true_labels = test_dataset["label"].numpy()
        pred_labels = predictions.predictions.argmax(axis=1)

        report = classification_report(true_labels, pred_labels,
                                        target_names=[id2label[i] for i in range(len(labels))])
        print("Classification Report:\n", report)
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        save_confusion_matrix(true_labels, pred_labels, id2label, out_dir)
        logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
