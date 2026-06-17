# SPAM_CLASSIFIER

Simple spam classifier project containing CNN, SNN, and Transformer model wrappers,
data preprocessing utilities, and notebook demos.

## Structure
- `CNN_module/` — CNN model and training notebook
- `SNN_module/` — SNN model and related scripts
- `Transformer_module/` — Transformer model and training notebook
- `data/` — datasets (not included in repo by default)
- `rf_email_spam_model.pkl`, `*.pth`, `*.safetensors` — model binaries (downloaded from GitHub Releases)

## Quick start
1. Create a Python venv and install dependencies (PyTorch, snntorch, transformers, scikit-learn, joblib).
2. Download the model files (`*.pth`, `*.safetensors`, `rf_email_spam_model.pkl`) from the latest GitHub Release (or use `scripts/download_models.py`).
3. Run `main.ipynb` or import `CNN_module.cnn_engine` / `SNN_module.snn_engine` / `Transformer_module.transformer_engine`.

## Notes
- Large dataset files and model binaries are intentionally ignored in the main branch. They are uploaded to GitHub Releases instead.
