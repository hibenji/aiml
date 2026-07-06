# AiML — Amazon Review Sentiment Analysis

Three notebooks comparing classic and deep learning approaches to sentiment
classification on Amazon "All Beauty" product reviews.

## Dataset

[`All_Beauty.jsonl.gz`](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz)
from the Amazon Reviews 2023 dataset (McAuley Lab, UCSD). Each notebook
downloads it automatically on first run and reuses the local copy after that.
Reviews are labeled positive/negative from star rating, balanced per class,
and split 75/25 into train/test sets.

## Notebooks

- **`AiML_naive_bayes.ipynb`** — TF-IDF features + Multinomial Naive Bayes baseline.
- **`AiML_logistic_regression.ipynb`** — TF-IDF features + Logistic Regression, with
  top-feature coefficient plots and review-length distribution analysis.
- **`AiML_neural_net.ipynb`** — Tokenized/padded sequences fed into a Bidirectional
  LSTM (Keras/TensorFlow) with early stopping.

All three log runs (metrics, confusion matrix, ROC curve) to
[Weights & Biases](https://wandb.ai) under the `aiml2026` entity, project
`amazon-sentiment-analysis`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn tensorflow wandb nltk nbformat
```

`wandb login` is required the first time you run a notebook, or set
`WANDB_MODE=offline` / `WANDB_MODE=disabled` to skip cloud logging.

## Usage

Open any notebook in Jupyter and run all cells top to bottom:

```bash
jupyter notebook
```
