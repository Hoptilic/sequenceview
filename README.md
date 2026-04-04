# Protein Function Classifier

This repository is centered on a single Google Colab notebook: [colab_training.ipynb](colab_training.ipynb).

It trains a PyTorch sequence classifier on UniProt amino acid sequences for a narrower binary task:

- `oxidoreductase` (EC 1.*)
- `hydrolase` (EC 3.*)

## Training Workflow

Open [colab_training.ipynb](colab_training.ipynb) in Google Colab and run Cell 2 with GPU enabled.

The notebook does everything end-to-end:

1. Installs dependencies (`requests`, `torch`).
2. Downloads reviewed UniProt entries for both classes.
3. Caches data to `/content/uniprot_binary_dataset.jsonl`.
4. Splits data with a sequence-identity-aware strategy to reduce leakage.
5. Trains a BiGRU-based classifier with AMP, gradient clipping, and early stopping.
6. Evaluates on a held-out test split.
7. Saves a checkpoint to Google Drive (if mounted) or local Colab storage.

## Outputs

- Per-epoch metrics: `train_loss`, `train_accuracy`, `validation_loss`, `validation_accuracy`
- Final test metrics dictionary
- Saved checkpoint path (default: `/content/drive/MyDrive/protein_classifier.pt`)
- Example inference output for a short amino acid sequence

## Notes

- This is a practical baseline for experimentation, not a production biology model.
- Labels are annotation-derived from UniProt queries and may include noise.
- If you change class counts or split parameters, rerun the training cell to regenerate cache and metrics.
