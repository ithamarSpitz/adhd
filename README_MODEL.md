---
sdk: gradio
title: ADHD Noise Classifier
python_version: "3.11"
app_file: app.py
---

# ADHD Classifier (Hugging Face Space)

This repository contains the minimal files to deploy the `adhd_model.pkl` scikit-learn model as a public Hugging Face Space using Gradio.

Files:
- `app.py` - Gradio app that loads `adhd_model.pkl` and exposes a JSON `/api/predict` endpoint.
- `adhd_model.pkl` - your trained model (should be uploaded by `upload_model_to_hf.py`).
- `requirements-hf.txt` - dependencies for the Space.
- `upload_model_to_hf.py` - script to create the Space and upload files programmatically.
- `update_model_on_hf.py` - quick script to re-upload an updated `adhd_model.pkl`.

Deploying (local, using the provided script)
1. Install uploader deps locally:
```powershell
python -m pip install --upgrade pip
pip install -r requirements-hf.txt
pip install huggingface_hub==0.18.1
```
2. Set your HF token in PowerShell:
```powershell
$Env:HF_TOKEN = Get-Content .\hf_token.txt
```
3. Run the upload script (choose a repo name):
```powershell
python upload_model_to_hf.py --model adhd_model.pkl --repo-name adhd-noise-classifier
```

Updating the model later:
```powershell
python update_model_on_hf.py --model adhd_model.pkl --repo-name adhd-noise-classifier
```

Calling the deployed API
- URL pattern: `https://huggingface.co/spaces/<username>/<repo>/api/predict`
- JSON body example:
```json
{"data": ["0.1,0.2,0.3,0.4"]}
```

Example Python client:
```python
import requests
url = "https://huggingface.co/spaces/<user>/adhd-noise-classifier/api/predict"
payload = {"data": ["0.1,0.2,0.3,0.4"]}
resp = requests.post(url, json=payload)
print(resp.json())
```

Notes:
- The Space will be public (free tier). If your model file grows beyond repo size limits, consider `git-lfs`.
- Do not share your `HF_TOKEN` in public. Keep it in `hf_token.txt` locally or as an environment variable.
