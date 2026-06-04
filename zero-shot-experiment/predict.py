#!/usr/bin/env python
# coding=utf-8
"""
Run genre classification predictions on a test CSV using a fine-tuned model.

Usage:
    python predict.py --model_dir output/train__xlmroberta-base \
                      --test_file data/prepared/UDM__lff__v3__chunked__balanced/test.csv \
                      --output_dir results/

Output:
    predictions.csv            – test DataFrame with added 'predicted_label' column
    classification_report.txt  – per-class precision / recall / F1
    confusion_matrix.png       – heatmap
"""

import os
import argparse
import logging

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_confusion_matrix(true_labels, pred_labels, class_names: list, output_dir: str):
    cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
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
    parser = argparse.ArgumentParser(description="Predict genre labels on a test CSV")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--test_file", required=True, help="Path to test CSV (with 'text' and 'label' columns)")
    parser.add_argument("--output_dir", default="results", help="Directory to save predictions and metrics")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model from {args.model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    id2label = model.config.id2label
    label2id = model.config.label2id
    labels = [id2label[i] for i in range(len(id2label))]

    logger.info(f"Loading test data from {args.test_file} …")
    test_df = pd.read_csv(args.test_file, index_col=0)
    test_df = test_df.dropna(subset=["text", "label"])

    dataset = ds.Dataset.from_pandas(test_df[["text", "label"]])
    label_feature = ds.ClassLabel(num_classes=len(labels), names=labels)
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

    # Map string labels to ids
    def encode_label(x):
        x["label"] = label2id.get(x["label"], -1)
        return x
    dataset = dataset.map(encode_label)
    dataset = dataset.cast_column("label", label_feature)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
        ),
    )

    logger.info("Running predictions …")
    preds_output = trainer.predict(dataset)
    true_ids = preds_output.label_ids
    pred_ids = preds_output.predictions.argmax(axis=1)

    true_labels = [id2label[i] for i in true_ids]
    pred_labels = [id2label[i] for i in pred_ids]

    # Save predictions
    test_df = test_df.copy()
    test_df["predicted_label"] = pred_labels
    test_df.to_csv(os.path.join(args.output_dir, "predictions.csv"))

    # Classification report
    report = classification_report(true_labels, pred_labels, target_names=labels)
    print("\nClassification Report:\n", report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    save_confusion_matrix(true_labels, pred_labels, labels, args.output_dir)

    logger.info(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
