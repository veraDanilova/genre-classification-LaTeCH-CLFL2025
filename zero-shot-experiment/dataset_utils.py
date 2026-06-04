"""
Dataset loading, preprocessing, balancing and splitting utilities
for CORE, FTD and UDM genre datasets.

Primary reference: actdisease-genre/zero-shot/label_mapping_&_preprocessing.ipynb
"""

import os
import re
import gzip
import lzma
import shutil
import requests
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mapping_utils import language_families, maps


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _remove_emoji(text: str) -> str:
    pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\ufe0f\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return pattern.sub("", text)


def clean_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for t in texts:
        t = re.sub(r"<a_href=\S+", "", t)
        t = re.sub(r"<(/)?\w+>", "", t)
        t = re.sub(r"[#*]+", "", t)
        t = re.sub(r"\[\s?\.+\s?\]", "", t)
        t = re.sub(r"/\w+/\s?|\(\s*ISBN.*?\)s?", "", t)
        t = _remove_emoji(t)
        cleaned.append(t)
    return cleaned


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> bool:
    print(f"Downloading {url} …")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"  ERROR {r.status_code}")
        return False
    text_exts = (".txt", ".tsv", ".csv", ".json")
    if any(dest.endswith(e) for e in text_exts):
        with open(dest, "w", encoding="utf-8") as f:
            f.write(r.text)
    else:
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return True


def _ungzip(src: str, dst: str):
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(src)


def _unlzma(src: str, dst: str):
    with lzma.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(src)


def download_CORE(data_dir: str):
    """Download CORE, FinCORE, SweCORE, FreCORE train/dev/test splits."""
    base = "https://raw.githubusercontent.com/TurkuNLP/"
    corpora = [
        ("FreCORE",  "Multilingual-register-corpora/main/data/FreCORE/", ".tsv"),
        ("SweCORE",  "Multilingual-register-corpora/main/data/SweCORE/", ".tsv"),
        ("FinCORE",  "FinCORE/master/data/",                             ".tsv"),
        ("CORE",     "CORE-corpus/master/",                              ".tsv.gz"),
    ]
    for name, path, ext in corpora:
        folder = os.path.join(data_dir, name)
        os.makedirs(folder, exist_ok=True)
        for split in ("train", "dev", "test"):
            url = base + path + split + ext
            dest = os.path.join(folder, split + ext)
            if _download(url, dest) and ext.endswith(".gz"):
                _ungzip(dest, dest.replace(".gz", ""))


def download_FTD(data_dir: str):
    """Download English and Russian FTD files."""
    base = "https://raw.githubusercontent.com/ssharoff/genre-keras/master/"
    folder = os.path.join(data_dir, "FTD")
    os.makedirs(folder, exist_ok=True)
    for fname in ("en.ol.xz", "en.gold.dat", "ru.ol.xz", "ru.csv"):
        dest = os.path.join(folder, fname)
        if _download(base + fname, dest):
            if fname == "en.gold.dat":
                _ungzip(dest, dest.replace(".dat", "_labels.dat"))
            elif fname.endswith(".xz"):
                _unlzma(dest, dest[:-3])  # strip .xz


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _expand_multi_label_rows(row: pd.Series, language: str, dataset: str) -> List[dict]:
    """Split space-separated multi-label strings into one row per label."""
    return [{"dataset": dataset, "language": language, "label": lbl, "text": row["text"]}
            for lbl in row["label"].split()]


def load_CORE(data_dir: str) -> pd.DataFrame:
    """Load and ensemble CORE + multilingual variants from data_dir."""
    folder = data_dir
    lang_map = {"Fin": "Finnish", "Swe": "Swedish", "Fre": "French"}
    frames = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isdir(path) or "CORE" not in name:
            continue
        language = "English" if name == "CORE" else lang_map[name.split("CORE")[0]]
        col_names = ["label", "id", "text"] if name == "CORE" else ["label", "text"]
        for split in ("train.tsv", "dev.tsv"):
            fpath = os.path.join(path, split)
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath, sep="\t", names=col_names, encoding="utf-8").dropna(subset=["label"])
            rows = [r for sub in df.apply(_expand_multi_label_rows, language=language, dataset=name, axis=1) for r in sub]
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def load_FTD(data_dir: str) -> pd.DataFrame:
    """Load English and Russian FTD from data_dir/FTD/."""
    folder = os.path.join(data_dir, "FTD")
    with open(os.path.join(folder, "en.ol"), encoding="utf-8") as f:
        en_texts = f.readlines()
    # File is named 'unzipped_en.gold.dat' after download, or 'en_labels.dat' if renamed
    label_file = next(
        (os.path.join(folder, n) for n in ("unzipped_en.gold.dat", "en_labels.dat", "en.gold.dat")
         if os.path.exists(os.path.join(folder, n))),
        None,
    )
    if label_file is None:
        raise FileNotFoundError(f"FTD English label file not found in {folder}")
    with open(label_file, encoding="utf-8") as f:
        en_labels = [re.sub(r"\n", "", l) for l in f.readlines()]
    with open(os.path.join(folder, "ru.ol"), encoding="utf-8") as f:
        ru_texts = f.readlines()
    ru_df = pd.read_csv(os.path.join(folder, "ru.csv"), index_col=0, sep="\t")
    ru_df["label"] = ru_df.loc[:, "A1":"A17"].idxmax(axis=1)
    return pd.DataFrame({
        "text":     ru_texts + en_texts,
        "language": ["Russian"] * len(ru_texts) + ["English"] * len(en_texts),
        "label":    ru_df["label"].tolist() + en_labels,
        "dataset":  ["FTD"] * (len(ru_texts) + len(en_texts)),
    })


def load_UDM(data_dir: str) -> pd.DataFrame:
    """Load Universal Dependencies Multi-Genre (UDM) corpus.

    Expects data_dir/UDM/ to contain genre subdirectories with UD treebank
    folders, each containing a *-train.txt file.

    Download instructions: https://github.com/UniversalDependencies
    Select treebanks tagged with genre metadata and place them under
    data_dir/UDM/<genre>/<UD_<Language>-<Treebank>/>.
    """
    udm_root = os.path.join(data_dir, "UDM")
    if not os.path.isdir(udm_root):
        raise FileNotFoundError(
            f"UDM data not found at {udm_root}. "
            "Download the UD-MULTIGENRE corpus and place it there."
        )
    rows = []
    for genre in os.listdir(udm_root):
        genre_path = os.path.join(udm_root, genre)
        if not os.path.isdir(genre_path):
            continue
        for root, _, files in os.walk(genre_path):
            train_files = [f for f in files if f.endswith("train.txt")]
            if not train_files:
                continue
            m = re.match(r"UD_([\w ]+)-[\w ]+", os.path.basename(root))
            language = m.group(1) if m else "Unknown"
            with open(os.path.join(root, train_files[0]), encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append({"label": genre, "language": language, "text": text, "dataset": "UDM"})
    return pd.DataFrame(rows)


def load_dataset(ds: str, data_dir: str) -> pd.DataFrame:
    loaders = {"CORE": load_CORE, "FTD": load_FTD, "UDM": load_UDM}
    if ds not in loaders:
        raise ValueError(f"Unknown dataset '{ds}'. Choose from: {list(loaders)}")
    return loaders[ds](data_dir)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def map_labels(df: pd.DataFrame, ds: str, mapping: str) -> pd.DataFrame:
    """Add 'cl_label' column with unified genre labels."""
    cl_map = maps[mapping]

    def _map(row):
        lbl = row["label"]
        result = [k for k in cl_map if lbl in cl_map[k].get(ds, [])]
        return result[0] if result else None

    df = df.rename(columns={"label": "original_label"})
    df["label"] = df.apply(_map, axis=1)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaNs, deduplicate on text, map language families, clean texts."""
    df = df.dropna().drop_duplicates(subset=["text"]).copy()
    df["language_family"] = df["language"].apply(
        lambda l: next((k for k, langs in language_families.items() if l in langs), None)
    )
    df["text"] = clean_texts(df["text"].tolist())
    return df.reset_index(drop=True)


def chunk_and_filter(
    df: pd.DataFrame,
    model_name: str = "FacebookAI/xlm-roberta-base",
    chunk_size: int = 490,
    chunk_overlap: int = 20,
    remove_punctuation: bool = False,
    remove_short: bool = True,
    language_family_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Chunk long texts, optionally filter by language family and strip punctuation."""
    if language_family_filter:
        df = df.query("language_family in @language_family_filter").copy()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def span_len(text):
        return len(tokenizer.encode(text))

    df["ntokens"] = df["text"].apply(span_len)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=span_len,
        separators=[".", ";", ",", ":", '"'],
    )

    split_rows = []
    for _, row in df.iterrows():
        if row["ntokens"] // 2 >= (512 - 2):
            for chunk in splitter.split_text(row["text"]):
                split_rows.append({**row.to_dict(), "text": chunk})
        else:
            split_rows.append(row.to_dict())

    df = pd.DataFrame(split_rows)
    df["text"] = df["text"].apply(lambda t: re.sub(r"^[\W\s_]*", "", t))

    if remove_punctuation:
        df["text"] = df["text"].apply(lambda t: re.sub(r"[^a-zA-Z0-9 \-]", "", t))

    if remove_short:
        df = df[df["text"].apply(lambda t: len(t.split()) > 5)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_dataset(df: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Two-level balancing: cap each top-level label at the median count,
    distributing the budget equally across its sublabels (combined_label).

    If 'combined_label' is absent, one-level balancing is applied.
    """
    df = df.copy()
    has_sublabel = "combined_label" not in df.columns

    if not has_sublabel:
        # Add combined label for two-level balancing
        df["combined_label"] = df["label"].astype(str) + "_" + df.get("original_label", df["label"]).astype(str)

    label_col = "label"
    threshold_genre = int(df[label_col].value_counts().median())
    result_rows = []

    for genre, group in df.groupby(label_col):
        if "combined_label" in df.columns:
            sub_col = "combined_label"
            n_subs = group[sub_col].nunique()
            budget = threshold_genre if len(group) > threshold_genre else len(group)
            per_sub = max(1, round(budget / n_subs))
            for _, subgroup in group.groupby(sub_col):
                sample = subgroup.sample(n=min(len(subgroup), per_sub),
                                         replace=False, random_state=random_state)
                result_rows.append(sample)
        else:
            n = min(len(group), threshold_genre)
            result_rows.append(group.sample(n=n, replace=False, random_state=random_state))

    return pd.concat(result_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def generate_dataset(
    ds: str,
    data_dir: str,
    mapping: str = "cl_map_v3",
    language_family_filter: Optional[List[str]] = None,
    chunking: bool = True,
    model_name: str = "FacebookAI/xlm-roberta-base",
    remove_punctuation: bool = False,
    remove_short: bool = True,
    balance: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load → map → preprocess → chunk/filter → balance → split."""
    df = load_dataset(ds, data_dir)
    df = map_labels(df, ds, mapping)
    df = basic_preprocessing(df)

    if chunking:
        df = chunk_and_filter(
            df,
            model_name=model_name,
            language_family_filter=language_family_filter,
            remove_punctuation=remove_punctuation,
            remove_short=remove_short,
        )
    elif language_family_filter:
        df = df.query("language_family in @language_family_filter").copy()

    if balance:
        df = balance_dataset(df, random_state=random_state)

    return split_train_test(df, test_size=test_size, random_state=random_state)
