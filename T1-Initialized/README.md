# BART Multi-Task Scientific Document Generation

This repository contains a modular implementation of the supplied BART multi-task
training pipeline for four generation tasks.

## Tasks

| Task ID | Input | Target |
|---:|---|---|
| 0 | Full article without introduction | Introduction |
| 1 | Full article without introduction | Abstract |
| 2 | Introduction | Abstract |
| 3 | Abstract | Title |

## Project structure

```text
.
├── bart_multitask/
│   ├── __init__.py
│   ├── config.py
│   ├── datasets.py
│   ├── collator.py
│   ├── losses.py
│   ├── trainer.py
│   └── modeling.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Dataset format

The CSV file must contain these columns:

- `article_text_no_intro`
- `abstract_text`
- `title`
- `introduction_text`

The training script creates an article-level 90/10 train/validation split using
random seed 42. The validation split is reused for both training stages.

## Configuration

Set paths using environment variables:

```bash
# Linux/macOS
export BART_MODEL_PATH="./pretrained/bart_large"
export BART_DATASET_PATH="./data/train_dataset.csv"
export BART_OUTPUT_DIR="./outputs"
```

```powershell
# Windows PowerShell
$env:BART_MODEL_PATH = ".\pretrained\bart_large"
$env:BART_DATASET_PATH = ".\data\train_dataset.csv"
$env:BART_OUTPUT_DIR = ".\outputs"
```

You can also edit the default paths in `bart_multitask/config.py`.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train

Run from the repository root:

```bash
python train.py
```

Training has two sequential stages:
1. Task-0 warm-up (introduction generation).
2. Multi-task training over all four tasks.

Checkpoints and the final model are saved under `outputs/`.

## Evaluate

```bash
python evaluate.py
```

The evaluation script reports the validation loss for the saved model. The
original supplied script did not include a separate test set or generation-based
ROUGE/BLEU/BERTScore evaluation implementation; those should be added when the
corresponding test data and evaluation protocol are available.

## Notes

- Do not upload private datasets, credentials, access tokens, or large model
  weights unless you have permission and intend to distribute them.
- The pretrained BART checkpoint is not included in this repository.
- The implementation preserves the original two-stage training design and
  task definitions while moving paths and settings into a configuration module.
