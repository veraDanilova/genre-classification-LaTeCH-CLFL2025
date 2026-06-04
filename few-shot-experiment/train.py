#!/usr/bin/env python
# coding=utf-8
"""
Few-shot genre classifier fine-tuning on the ActDisease corpus.

Loads a domain-adapted checkpoint (from MLM-finetuning/), trains a sequence
classifier, and evaluates on the held-out test portion of the dev file.

Usage:
    python train.py configs/train_xlmroberta.yaml --train_file 100__.train.csv

GPU selection:
    CUDA_VISIBLE_DEVICES=0 python train.py configs/train_xlmroberta.yaml --train_file 100__.train.csv

The dev file is split 50/50 into validation (used during training) and
test (used for the final classification report).
"""

import os
import logging
import yaml
import argparse

import torch
import datasets as ds
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"   # overridden below if wandb is enabled in config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_confusion_matrix(true_ids, pred_ids, id2label: dict, output_dir: str):
    labels = [id2label[i] for i in sorted(id2label)]
    cm = sklearn.metrics.confusion_matrix(true_ids, pred_ids)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML training config")
    parser.add_argument("--train_file", help="Override train_file from config (filename only, relative to data_dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Allow CLI override of train_file (for use in shell loops)
    if args.train_file:
        cfg["train_file"] = args.train_file

    set_seed(cfg.get("seed", 1234))

    checkpoint  = cfg["checkpoint"]
    data_dir    = cfg.get("data_dir", "data_few-shot-data")
    train_file  = cfg["train_file"]
    dev_file    = cfg.get("dev_file", "few_shot.dev.csv")
    output_root = cfg.get("output_dir", "output")

    # wandb setup
    if cfg.get("use_wandb", False):
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "few-shot-genre")
        import wandb
        wandb.login()
    else:
        os.environ["WANDB_DISABLED"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Derive a run name from data size + model
    run_name = os.path.splitext(train_file)[0] + "__" + os.path.basename(checkpoint)
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Run: {run_name}")
    logger.info(f"Checkpoint: {checkpoint}")

    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, train_file), index_col=0)
    dev_df   = pd.read_csv(os.path.join(data_dir, dev_file), index_col=0)

    labels   = sorted(train_df["label"].dropna().unique().tolist())
    id2label = dict(enumerate(labels))
    label2id = {l: i for i, l in id2label.items()}

    logger.info(f"Labels ({len(labels)}): {labels}")
    logger.info(f"Train size: {len(train_df)}")

    # Split dev → validation + test (50 / 50, stratified)
    valid_df, test_df = train_test_split(
        dev_df,
        test_size=0.5,
        stratify=dev_df["label"],
        random_state=cfg.get("seed", 1234),
    )

    logger.info(f"Valid: {len(valid_df)}  Test: {len(test_df)}")

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    label_feature = ds.ClassLabel(num_classes=len(labels), names=labels)

    def make_dataset(df: pd.DataFrame) -> ds.Dataset:
        df = df[["text", "label"]].copy()
        df["label"] = df["label"].map(label2id)
        dataset = ds.Dataset.from_pandas(df)
        dataset = dataset.cast_column("label", label_feature)
        dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    train_dataset = make_dataset(train_df)
    valid_dataset = make_dataset(valid_df)
    test_dataset  = make_dataset(test_df)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=cfg.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=cfg.get("attention_probs_dropout_prob", 0.25),
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.get("learning_rate", 1e-5),
        per_device_train_batch_size=cfg.get("batch_size", 8),
        per_device_eval_batch_size=cfg.get("batch_size", 8),
        num_train_epochs=cfg.get("num_epochs", 5),
        weight_decay=cfg.get("weight_decay", 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=run_name,
        seed=cfg.get("seed", 1234),
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

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Predict on held-out test set
    preds_output = trainer.predict(test_dataset)
    true_ids = preds_output.label_ids
    softmax  = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, preds_output.predictions)
    pred_ids = softmax.argmax(axis=1)

    report = classification_report(
        true_ids, pred_ids,
        target_names=[id2label[i] for i in range(len(labels))],
    )
    print("\nClassification Report:\n", report)

    with open(os.path.join(out_dir, "cl_report.txt"), "w") as f:
        for i, l in id2label.items():
            f.write(f"{i}: {l}\n")
        f.write(report)

    save_confusion_matrix(true_ids, pred_ids, id2label, out_dir)

    # Save test predictions
    result_df = test_df[["language", "label", "text"]].copy().reset_index(drop=True)
    result_df["pred"]  = [id2label[i] for i in pred_ids]
    result_df["score"] = softmax.max(axis=1)
    result_df.to_csv(os.path.join(out_dir, "test_predictions.csv"))

    logger.info(f"All results saved to {out_dir}")


if __name__ == "__main__":
    main()
