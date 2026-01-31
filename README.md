# ADHD Game Session Noise Classifier

Train and predict noise in classroom game sessions.

## Setup

**ADHD Game Session Noise Classifier**

- **Purpose:** Train a scikit-learn classifier to detect noisy game sessions and host it as an HTTP API on a Hugging Face Space.

**Files of Interest**
- `game_dataset.csv` : example dataset used for training and tests.
- `train_noise_model.py` : training script that saves a scikit-learn pipeline to a `.pkl` file.
- `noise_model.pkl` : the serialized model file (created by training).
- `app.py` : the Space entrypoint — FastAPI app exposing `POST /predict` for programmatic access.
- `requirements-hf.txt` : packages installed by the Space during build.
- `upload_model_to_hf.py` / `update_model_on_hf.py` : helpers to create/update the Space repository and upload files.
- `call_predict.py` / `call_subdomain_api.ps1` : example client scripts for calling the deployed API.
- `get_space_logs.py`, `get_runtime.py` : helper scripts to inspect Space runtime and logs.

**Requirements**
- Create a Python environment and install locally for development:
	```bash
	pip install -r requirements-hf.txt
	```

**Train & Save the Model (local)**
- Train using your CSV and save the pipeline to `noise_model.pkl`:
	```bash
	python train_noise_model.py --data game_dataset.csv --model noise_model.pkl
	```
- Confirm the saved model exposes `n_features_in_` (the number of features). The Space expects inputs with the same number of features (this model expects 23 features).

**Upload / Deploy to Hugging Face Spaces**
- You can create or update the Space using the provided scripts. The scripts read a token from the `HF_TOKEN` environment variable or `hf_token.txt`.
- Create / upload (first time):
	```bash
	# set token in env (recommended) or ensure hf_token.txt contains it
	setx HF_TOKEN "<your-token>"          # Windows PowerShell, or use $Env:HF_TOKEN for session
	python upload_model_to_hf.py --model noise_model.pkl
	```
- Update only the model file (faster):
	```bash
	python update_model_on_hf.py --model noise_model.pkl
	```
- Notes:
	- `requirements-hf.txt` is used by the Space builder; we added `fastapi` and `uvicorn` so the FastAPI app runs.
	- If you prefer a Gradio UI, revert `app.py` to the Gradio interface and ensure `gradio` is listed in `requirements-hf.txt`.

**Call the Model via API (examples)**
- Endpoint (example for this deployment):
	- `https://ithamarspitz-adhd-noise-classifier.hf.space/predict`

- Payload format (JSON):
	- Single row: `{"data": [[v1, v2, ..., v23]]}`
	- Batch: `{"data": [[row1_vals...], [row2_vals...]]}`

- cURL (Linux/macOS / WSL):
	```bash
	curl -X POST "https://ithamarspitz-adhd-noise-classifier.hf.space/predict" \
		-H "Content-Type: application/json" \
		-d '{"data": [[56,22,2,3,1,81,2.41,0.76,212,1.36,1.84,4,0,0,18,133,0.324,1.498,2,0.585,1.285,6,6]]}'
	```

- PowerShell (Windows):
	```powershell
	$payload = @{ data = @( @(56,22,2,3,1,81,2.41,0.76,212,1.36,1.84,4,0,0,18,133,0.324,1.498,2,0.585,1.285,6,6) ) }
	Invoke-RestMethod -Method Post -Uri 'https://ithamarspitz-adhd-noise-classifier.hf.space/predict' -ContentType 'application/json' -Body ($payload | ConvertTo-Json)
	```

- Python `requests`:
	```python
	import requests
	url = 'https://ithamarspitz-adhd-noise-classifier.hf.space/predict'
	payload = {"data": [[56,22,2,3,1,81,2.41,0.76,212,1.36,1.84,4,0,0,18,133,0.324,1.498,2,0.585,1.285,6,6]]}
	r = requests.post(url, json=payload, timeout=30)
	print(r.status_code, r.json())
	```

**How to update the model workflow (full)**
1. Retrain locally and overwrite `noise_model.pkl`:
	 ```bash
	 python train_noise_model.py --data game_dataset.csv --model noise_model.pkl
	 ```
2. Validate locally with `test.py` or `call_predict.py`.
3. Upload the new model to the existing Space:
	 ```bash
	 python update_model_on_hf.py --model noise_model.pkl
	 ```
4. Wait for the Space to rebuild (watch the Space logs on the Hugging Face UI) and then call the `/predict` endpoint.

**Troubleshooting & Logs**
- Check Space status and runtime:
	```bash
	python get_runtime.py
	```
- Stream container logs (if you have appropriate token scopes):
	```bash
	python get_space_logs.py
	```
- Common issues:
	- 403 on job listing: token lacks jobs/log scopes.
	- 503 / 504: Space is rebuilding or crashed — check logs for stack traces.
	- Feature-count mismatch: the model expects a specific number of features (see `model.n_features_in_`). Send the same number per row.

**Notes**
- The deployed entrypoint is a small FastAPI app (`/predict`). This provides a stable programmatic API and is easy to update via the `update_model_on_hf.py` helper.
- If you want a web UI for interactive testing, I can add a small Gradio front-end while keeping the `/predict` API endpoint.

If you want, I can now:
- add a short `README_MODEL.md` model card with metrics, or
- run a batch test across rows in `game_dataset.csv` and summarize predictions.
