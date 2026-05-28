# SPAM_CLASSIFIER

Simple spam classifier project containing CNN and Transformer model wrappers,
data preprocessing utilities, and notebook demos.

## Structure
- `CNN_module/` — CNN model and training notebook
- `Transformer_module/` — Transformer model and training notebook
- `data/` — datasets (not included in repo by default)
- `rf_email_spam_model.pkl`, `*.pth` — model binaries (ignored by `.gitignore`)

## Quick start
1. Create a Python venv and install dependencies (PyTorch, transformers, scikit-learn, joblib).
2. Place model files (`*.pth`, `rf_email_spam_model.pkl`) in the repo root or `Transformer_module/my_bert_model`.
3. Run `main.ipynb` or import `CNN_module.cnn_engine` / `Transformer_module.transformer_engine`.

## Notes
- Large model and dataset files are intentionally ignored. Use Git LFS or upload models separately.
