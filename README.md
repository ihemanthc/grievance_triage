<div align="center">

# Grievance Triage

### A practical NLP pipeline for classifying public grievances by category and urgency

<p>
  <a href="https://colab.research.google.com/github/ihemanthc/grievance_triage/blob/main/grievance_triage_notebook.ipynb">
    <img src="https://img.shields.io/badge/Open%20in%20Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Google Colab">
  </a>
  <a href="https://github.com/ihemanthc/grievance_triage/blob/main/grievance_triage_notebook.ipynb">
    <img src="https://img.shields.io/badge/View%20Notebook-181717?style=for-the-badge&logo=github&logoColor=white" alt="View notebook on GitHub">
  </a>
  <a href="https://github.com/ihemanthc/grievance_triage/tree/main/outputs">
    <img src="https://img.shields.io/badge/View%20Outputs-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="View generated outputs">
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Transformers">
</p>

</div>

## Overview

This project provides a reproducible notebook for the **Grievance Triage Challenge**. Given a public grievance, the pipeline predicts two independent attributes:

- **Category** — one of eight grievance categories
- **Urgency** — `routine`, `high`, or `critical`

The final prediction is written in the competition format:

```text
category|urgency
```

The approach deliberately separates category and urgency into two classifiers. This reflects the analysis in the notebook: category is highly predictable from the text, while urgency requires more attention to the actual severity of the complaint.

## Approach

### TF-IDF baseline

The baseline combines:

- Word-level TF-IDF features using 1–2 grams
- Character-level TF-IDF features using 3–5 character n-grams
- One-hot encoded `complaint_history`
- Two balanced `LinearSVC` classifiers: one for category and one for urgency

Character features improve robustness to spelling variation and code-mixed language. `channel` and `district` are excluded because the notebook’s analysis found little useful signal in those fields.

### Optional transformer model

The notebook also includes an optional GPU workflow that fine-tunes [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased) for urgency classification. MuRIL is a useful fit for Indian-language and transliterated text. The transformer urgency prediction can be combined with the TF-IDF category prediction when it improves validation performance.

## Quick start

1. Open the notebook in Google Colab using the **Open in Google Colab** button above.
2. For the transformer section, select **Runtime → Change runtime type → GPU**. GPU is optional for the TF-IDF baseline.
3. Run the notebook cells in order.
4. When prompted, upload `train.csv`, `test.csv`, and `sample_submission.csv`.
5. Download the generated submission file from Colab.

The baseline section produces `submission_baseline_tfidf.csv`. The optional transformer section produces `submission_transformer.csv`.

## Repository structure

```text
.
├── grievance_triage_notebook.ipynb   # End-to-end training and inference notebook
├── inputs/
│   ├── train.csv                     # Training data
│   ├── test.csv                      # Test data
│   └── sample_submission.csv         # Required submission schema
├── outputs/                           # Saved submission examples
└── README.md
```

## Data format

The notebook expects the following fields:

| File | Required fields |
| --- | --- |
| `train.csv` | `id`, `subject`, `body`, `complaint_history`, `label` |
| `test.csv` | `id`, `subject`, `body`, `complaint_history` |
| `sample_submission.csv` | `id`, `label` |

The training `label` is split into `category` and `urgency` using the `category|urgency` format.

## Reproducibility notes

- The cross-validation split uses five stratified folds with `random_state=42`.
- The baseline is CPU-friendly and is the fastest way to generate a submission.
- The transformer workflow requires additional packages and is best run with a GPU.
- Model outputs can vary slightly across hardware and library versions.
