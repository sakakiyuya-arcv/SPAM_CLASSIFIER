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
1. Clone the repository and navigate into the project directory.
2. Create a Python virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets (`*.csv`) from the latest GitHub Release and place them inside the `data/` folder (Create this folder if it doesn't exist).
4. Download the model files from the latest GitHub Release:
   - Place `cnn_model.pth` in `CNN_module/`
   - Place `snn_model.pth` and `tfidf_vectorizer.pkl` in `SNN_module/`
   - Place `transformer_model.pth` in `Transformer_module/`
   - **Important**: Download `model.safetensors`, `config.json`, `vocab.txt`, and `tokenizer_config.json` and place them all inside a new folder named `Transformer_module/my_bert_model/` (Create this folder if it doesn't exist).
   - Place `rf_email_spam_model.pkl` in the root folder.
5. Run `main.ipynb` or import `CNN_module.cnn_engine` / `SNN_module.snn_engine` / `Transformer_module.transformer_engine`.

## Notes
- Large dataset files and model binaries are intentionally ignored in the main branch. They are uploaded to GitHub Releases instead.
