# Classifying Textual Genre in Historical Magazines (1875–1990)

Code for the paper **"Classifying Textual Genre in Historical Magazines (1875–1990)"** presented at LaTeCH-CLfL 2025 (co-located with ACL 2025, Albuquerque, New Mexico).

> Danilova, Vera and Söderfeldt, Ylva. "Classifying Textual Genre in Historical Magazines (1875-1990)." *Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025)*, pp. 160–171. ACL, 2025. https://aclanthology.org/2025.latechclfl-1.15/

## Overview

Historical magazines often combine texts of diverse communicative purposes — personal narratives, advertisements, instructions, short stories — on the same page. Without grouping documents by genre, downstream analyses such as term counts or topic models may yield misleading results.

This repository provides code for genre classification within a digitised collection of European medical magazines in **Swedish and German** (1875–1990) from the **ActDisease corpus** (ERC-2021-STG 10104099). Two experimental scenarios are explored:

1. **Zero-shot** — train on publicly available web-genre datasets (CORE, FTD, UDM) and predict directly on ActDisease without any in-domain labelled data.
2. **Few-shot / semi-supervised** — fine-tune on small labelled subsets (100–500 samples) of ActDisease and evaluate on a fixed held-out test set.

Both scenarios use multilingual BERT-family models optionally adapted to the ActDisease domain via continued masked language model (MLM) pre-training.

**Genre labels:** `QA`, `academic`, `administrative`, `advertisement`, `fiction`, `non_fiction`

**Models:** XLM-RoBERTa (`FacebookAI/xlm-roberta-base`), hmBERT (`dbmdz/bert-base-historic-multilingual-cased`), mBERT (`google-bert/bert-base-multilingual-cased`)

**Key findings:**
- A custom genre scheme bridging web and historical-magazine categories enables effective cross-domain, cross-lingual zero-shot prediction.
- Semi-supervised training gives considerable gains over few-shot for all models, especially for hmBERT.
- Domain-adaptive MLM pre-training on ActDisease consistently improves classification.

> **Data availability:** The ActDisease corpus is not publicly available due to privacy constraints. Contact the repository author for access.

## Repository Structure

```
genre-classification-LaTeCH-CLFL2025/
├── MLM-finetuning/          # Domain-adaptive MLM pre-training on ActDisease
├── zero-shot-experiment/    # Zero-shot classification via web-genre datasets
├── few-shot-experiment/     # Few-shot / semi-supervised classification
└── requirements.txt         # Shared Python dependencies
```

Each subdirectory contains its own `README.md` with detailed usage instructions.

## Setup

Requires Python ≥ 3.9 and a CUDA-capable GPU (recommended).

```bash
git clone https://github.com/...
cd genre-classification-LaTeCH-CLFL2025
pip install -r requirements.txt
```

## Reproducing the Experiments

The three pipelines are independent but feed into each other in the order below.

---

### 1. MLM Fine-tuning (`MLM-finetuning/`)

Domain-adapts three multilingual models to the ActDisease corpus via continued MLM pre-training.

```bash
cd MLM-finetuning

# Step 1 – preprocess the corpus into line-by-line text files
python prepare_data.py configs/prepare_data.yaml

# Step 2 – fine-tune each model (select GPU via CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/hmbert.json      # 20 epochs
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/mbert.json       #  5 epochs
CUDA_VISIBLE_DEVICES=0 python run_mlm.py configs/xlmroberta.json  # 10 epochs
```

Checkpoints are saved to `output/{hmbert,mbert,xlmroberta}-ft-mlm/`.

See [`MLM-finetuning/README.md`](MLM-finetuning/README.md) for config options and output details.

---

### 2. Zero-Shot Experiment (`zero-shot-experiment/`)

Trains classifiers on web-genre datasets and predicts directly on the ActDisease test set.

```bash
cd zero-shot-experiment

# Step 1 – prepare one or more training datasets
python prepare_dataset.py configs/datasets/CORE.yaml
python prepare_dataset.py configs/datasets/FTD.yaml
python prepare_dataset.py configs/datasets/UDM.yaml
python prepare_dataset.py configs/datasets/merged.yaml   # CORE + FTD + UDM

# Step 2 – fine-tune a classifier on the prepared data
CUDA_VISIBLE_DEVICES=0 python train.py configs/training/xlmroberta.yaml
CUDA_VISIBLE_DEVICES=0 python train.py configs/training/hmbert.yaml
CUDA_VISIBLE_DEVICES=0 python train.py configs/training/mbert.yaml

# Step 3 – predict on the ActDisease test set
python predict.py \
    --model_dir output/train__xlmroberta-base/best_model \
    --test_file path/to/actdisease_test.csv \
    --output_dir results/xlmroberta_UDM
```

> UDM (Universal Dependencies Multi-Genre) must be obtained manually from https://universaldependencies.org/ and placed under `data/zero-shot_data/UDM/<genre>/<UD_Language-Treebank>/`.

See [`zero-shot-experiment/README.md`](zero-shot-experiment/README.md) for dataset configs, label mapping versions, and the two-level balancing strategy.

---

### 3. Few-Shot Experiment (`few-shot-experiment/`)

Fine-tunes classifiers on labelled ActDisease subsets (100–500 samples) using MLM-adapted checkpoints from step 1.

```bash
cd few-shot-experiment

# Step 1 – generate stratified N-shot splits (pre-generated splits already included)
python prepare_data.py configs/prepare_data.yaml

# Step 2 – run all models × all data sizes
CUDA_VISIBLE_DEVICES=0 bash run_experiments.sh

# Or a single model on all sizes:
CUDA_VISIBLE_DEVICES=0 bash run_experiments.sh configs/train_hmbert.yaml

# Or a single model on a specific size:
CUDA_VISIBLE_DEVICES=0 python train.py configs/train_hmbert.yaml --train_file 100__.train.csv
```

Results are written to `output/<train_file>__<checkpoint_name>/` — one subdirectory per run containing a classification report, confusion matrix, and per-sample predictions.

See [`few-shot-experiment/README.md`](few-shot-experiment/README.md) for config options and output details.

---

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
