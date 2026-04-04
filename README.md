# Protein Function Classifier

This project downloads labeled protein sequences from UniProt and trains a small PyTorch classifier to distinguish enzymes from non-enzymes.

The label is derived from UniProt annotations:

- `enzyme`: reviewed UniProt entries with an EC number
- `non-enzyme`: reviewed UniProt entries without an EC number

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 train.py
```

If you want to train in Google Colab, open [colab_training.ipynb](colab_training.ipynb) and run it with GPU enabled.

The script will:

1. Download the sequences from UniProt.
2. Cache them under `data/`.
3. Train a simple CNN classifier in PyTorch.
4. Save the checkpoint under `artifacts/`.

## Predict

```bash
python3 train.py --predict MVLSPADKTNVKAAW
```

## Tuning

Useful flags:

- `--enzyme-count` and `--non-enzyme-count` to control dataset size
- `--max-length` to control truncation/padding length
- `--epochs` and `--batch-size` for training speed/quality tradeoffs
- `--refresh-data` to force a new UniProt download

## Notes

This is a lightweight baseline, not a production biology model. The labels are annotation-based and the classifier is intentionally simple so it stays easy to run and inspect.
