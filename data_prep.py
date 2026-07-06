"""Shared data preparation for the Amazon All-Beauty sentiment task.

Everything up to and including the train/test split is identical no matter
which model consumes it, so it lives here and both notebooks import it. This
guarantees the neural network and the logistic-regression model are trained
and evaluated on the *exact same* reviews, labels, and split.

Model-specific feature engineering stays in each notebook, because the two
models need fundamentally different inputs:
    - neural net        -> Keras Tokenizer + pad_sequences (integer sequences)
    - logistic regression -> cleaning + lemmatization + TF-IDF (bag of words)
"""

import os
import gzip
import json
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
       "raw/review_categories/All_Beauty.jsonl.gz")
FILE_NAME = "All_Beauty.jsonl.gz"


def download_dataset(url=URL, file_name=FILE_NAME):
    """Download the dataset once; skip if it is already present."""
    if os.path.exists(file_name):
        print(f"'{file_name}' already exists, skipping download.")
    else:
        print("Downloading dataset (this may take a moment)...")
        urllib.request.urlretrieve(url, file_name)
        print("Download complete.")
    return file_name


def load_raw_dataframe(file_name=FILE_NAME):
    """Parse the full JSONL.GZ file into a DataFrame."""
    download_dataset(file_name=file_name)
    print("Parsing JSONL.GZ file...")
    data = []
    with gzip.open(file_name, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} records.")
    return df


def load_review_data(text_field="full_text", max_per_class=None,
                     test_size=0.25, seed=42):
    """Download, label, balance, and split the reviews.

    Parameters
    ----------
    text_field : which text column to feed the models. Default "full_text"
        (title + text) gives the models maximum context; both models use it,
        so they stay comparable.
    max_per_class : optional cap on reviews per class (use for a faster run).
        Default None uses every available review after balancing.
    test_size, seed : passed to a stratified train/test split.

    Returns
    -------
    dict with keys:
        df       : the balanced, shuffled DataFrame (for the EDA plots)
        X_train, X_test, y_train, y_test : the canonical split (same for every model)
        counts   : raw + balanced class counts for the distribution charts
    """
    df = load_raw_dataframe()

    # Combine title + text for maximum context (shared choice for both models)
    df = df.copy()
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[df["full_text"].str.strip() != ""]

    # Raw class counts, captured before we drop anything (for the "before" chart)
    raw_neg = int(df["rating"].isin([1.0, 2.0]).sum())
    raw_neu = int((df["rating"] == 3.0).sum())
    raw_pos = int(df["rating"].isin([4.0, 5.0]).sum())

    # Binary labels; 3-star (neutral) reviews are dropped
    df = df[df["rating"] != 3.0]
    df["sentiment"] = (df["rating"] >= 4).astype(int)

    # Balance the two classes by downsampling the majority
    positives = df[df["sentiment"] == 1]
    negatives = df[df["sentiment"] == 0]
    per_class = min(len(positives), len(negatives))
    if max_per_class is not None:
        per_class = min(per_class, max_per_class)

    pos_bal = resample(positives, replace=False, n_samples=per_class, random_state=seed)
    neg_bal = resample(negatives, replace=False, n_samples=per_class, random_state=seed)

    balanced = (pd.concat([pos_bal, neg_bal])
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    # The one canonical split every model reuses
    X = balanced[text_field]
    y = balanced["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    counts = {
        "raw_negative": raw_neg,
        "raw_neutral": raw_neu,
        "raw_positive": raw_pos,
        "dropped_3_star": raw_neu,
        "balanced_per_class": per_class,
        "balanced_total": 2 * per_class,
    }

    print(f"\nDataset ready: {counts['balanced_total']:,} balanced reviews "
          f"({per_class:,} per class); {raw_neu:,} neutral (3-star) dropped.")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    return {
        "df": balanced,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "counts": counts,
    }
