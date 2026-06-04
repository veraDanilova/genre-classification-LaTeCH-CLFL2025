# Few-Shot Genre Classification Experiment

Semi-supervised few-shot genre classification on the ActDisease corpus of historical European medical magazines (German and Swedish, 1875–1990). Part of the paper **"Classifying Textual Genre in Historical Magazines (1875–1990)"** (LaTeCH-CLfL 2025).

## Overview

Classifiers are trained on small labelled subsets (100–500 samples) of the ActDisease corpus and evaluated on a fixed held-out test set. Three domain-adapted models are compared, each first pre-trained on the full ActDisease text via masked language modelling (see `MLM-finetuning/`).

**Genre labels:** `QA`, `academic`, `administrative`, `advertisement`, `fiction`, `non_fiction`
**Languages:** German (`deu`), Swedish (`swe`)

**Models fine-tuned:**

| Config | Base model |
|--------|-----------|
| `configs/train_xlmroberta.yaml` | `FacebookAI/xlm-roberta-base` (MLM-adapted) |
| `configs/train_hmbert.yaml` | `dbmdz/bert-base-historic-multilingual-cased` (MLM-adapted) |
| `configs/train_mbert.yaml` | `google-bert/bert-base-multilingual-cased` (MLM-adapted) |

**Training set sizes:** 100, 200, 300, 400, 500 samples, and the full labelled set.

## Repository Structure

```
few-shot-experiment/
├── prepare_data.py          # Generate few-shot splits from annotated CSV
├── train.py                 # Train and evaluate a genre classifier
├── run_experiments.sh       # Run all model × data-size combinations
├── configs/
│   ├── prepare_data.yaml    # Paths and split parameters
│   ├── train_xlmroberta.yaml
│   ├── train_hmbert.yaml
│   └── train_mbert.yaml
├── data_few-shot-data/      # Pre-generated splits (ready to use)
│   ├── few_shot.dev.csv     # Fixed held-out test/dev set (~507 samples)
│   ├── 100__.train.csv
│   ├── 200__.train.csv
│   ├── 300__.train.csv
│   ├── 400__.train.csv
│   ├── 500__.train.csv
│   └── all__.train.csv      # Full labelled training pool (~1182 samples)
├── output/                  # Model checkpoints and results (created at runtime)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The MLM-adapted checkpoints are required before training. Run the MLM fine-tuning pipeline first:

```bash
cd ../MLM-finetuning
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/hmbert.json
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/mbert.json
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/xlmroberta.json
```

## Usage

### Step 1 – Prepare data splits (optional)

The `data_few-shot-data/` splits are already provided. To regenerate them from a new annotated CSV:

```bash
python prepare_data.py configs/prepare_data.yaml
```

The input CSV must have columns `language`, `label`, `text` (same structure as `all__.train.csv`).
The script produces stratified N-shot subsets and saves them alongside the dev file.

To create splits from a fresh annotation (no existing dev set):

```yaml
# configs/prepare_data.yaml
input_file: path/to/annotated.csv
test_file: null          # will be split automatically using test_size
test_size: 0.3
```

### Step 2 – Train

**Run all models on all data sizes:**

```bash
CUDA_VISIBLE_DEVICES=0 bash run_experiments.sh
```

**Run a single model on all data sizes:**

```bash
CUDA_VISIBLE_DEVICES=0 bash run_experiments.sh configs/train_hmbert.yaml
```

**Run a single model on a specific data size:**

```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/train_hmbert.yaml --train_file 100__.train.csv
```

Each run produces a dedicated output directory:

```
output/<train_file>__<checkpoint_name>/
    best_model/                  # saved model + tokenizer
    cl_report.txt                # classification report (label mapping + per-class metrics)
    confusion_matrix.png         # heatmap
    test_predictions.csv         # per-sample predictions with confidence scores
```

### Weights & Biases logging

Set `use_wandb: true` in any training config and set `wandb_project` to your project name. The first run will prompt for a wandb login, or set `WANDB_API_KEY` in your environment.

## Data Format

All CSVs share the same structure:

| Column | Description |
|--------|-------------|
| `language` | ISO 639-1 code: `deu` or `swe` |
| `label` | Genre label |
| `text` | Paragraph-level text |

The dev file (`few_shot.dev.csv`) is split 50/50 into validation (used during training) and final test (used for the classification report) inside `train.py`.

## Data Provenance

The ActDisease corpus was digitised from physical magazines held by European patient organisations. Raw scans were OCR'd to XML and parsed into paragraph-level blocks using `parser_xml.ipynb`. Genre labels were assigned manually. For privacy reasons the full corpus is not publicly available; contact the repository author for access.

The unlabelled text (used for MLM pre-training) was produced by `prepare_for_mlm_ft.ipynb`, which aggregates parsed paragraph blocks by issue and saves them to `actdisease_dataset.csv`.

## Configuration Reference

### `configs/prepare_data.yaml`

| Key | Description | Default |
|-----|-------------|---------|
| `input_file` | Annotated CSV (`language`, `label`, `text`) | required |
| `test_file` | Pre-existing dev/test CSV; if null, split from input | `null` |
| `output_dir` | Where to write splits | `data_few-shot-data` |
| `sizes` | Few-shot subset sizes | `[100,200,300,400,500]` |
| `test_size` | Fraction held out when `test_file` is null | `0.3` |
| `label_col` | Label column name | `label` |
| `random_state` | Random seed | `42` |

### `configs/train_*.yaml`

| Key | Description | Default |
|-----|-------------|---------|
| `checkpoint` | Path to MLM-finetuned model directory | required |
| `data_dir` | Directory containing train/dev CSVs | `data_few-shot-data` |
| `train_file` | Training CSV filename (overridden by `--train_file`) | `all__.train.csv` |
| `dev_file` | Dev/test CSV filename | `few_shot.dev.csv` |
| `output_dir` | Root for output subdirectories | `output` |
| `learning_rate` | AdamW learning rate | `1e-5` |
| `batch_size` | Per-device batch size | `8` |
| `num_epochs` | Training epochs | `5` |
| `weight_decay` | AdamW weight decay | `0.0` |
| `seed` | Random seed | `1234` |
| `hidden_dropout_prob` | Classifier head dropout | `0.1` |
| `attention_probs_dropout_prob` | Attention dropout | `0.25` |
| `use_wandb` | Enable W&B logging | `false` |
| `wandb_project` | W&B project name | – |

## Reference

```bibtex
@inproceedings{danilova-soderfeldt-2025-classifying,
    title = "Classifying Textual Genre in Historical Magazines (1875-1990)",
    author = {Danilova, Vera and S{\"o}derfeldt, Ylva},
    booktitle = "Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.latechclfl-1.15/",
    doi = "10.18653/v1/2025.latechclfl-1.15",
    pages = "160--171"
}
```
