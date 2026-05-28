# Adding Model Files

Recommended: use Git LFS to store model binaries (`*.pth`, `*.safetensors`, `*.pkl`). Steps:

1. Install Git LFS (Windows examples):

   - `winget install Git.GitLFS`  or
   - `choco install git-lfs` or
   - Download from https://git-lfs.github.com/

2. Enable LFS in the repo (once):

```powershell
git lfs install
```

3. Track model files (already configured in this repo):

```powershell
git lfs track "*.pth" "*.safetensors" "rf_email_spam_model.pkl"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

4. Add and push a model file:

```powershell
git add path\to\your\model.pth
git commit -m "Add CNN model via LFS"
git push origin main
```

Alternate options:
- Upload model files to a cloud bucket (S3, Google Drive) and provide download script under `scripts/`.
- Use GitHub Releases to attach pre-built model binaries.
