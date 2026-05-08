# Running TRIBE v2 on Modal GPUs

This repository includes support for offloading the heavy AI inference (video encoding, feature extraction, and prediction) to [Modal](https://modal.com/), a serverless GPU platform. 

This allows you to run the web application (FastAPI, reports, and visualization) locally on your laptop, while the massive 2-Billion parameter Vision Transformer (`vjepa2`) runs on a cloud A100 GPU.

## Setup Instructions

1. **Install Modal**
   Ensure you have the `modal` Python package installed in your local environment:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**
   Run the setup command to log in to your Modal account:
   ```bash
   modal setup
   ```

3. **Download the Models to Modal (One-time setup)**
   The AI models need to be downloaded into a persistent Modal Volume so they don't have to be re-downloaded every time the container starts. This takes about 20-30 minutes.
   ```bash
   modal run modal_inference.py::download_models
   ```

4. **Deploy the Inference Endpoint**
   Deploy the `TribeInference` class to Modal. This creates a persistent, serverless endpoint that your local app can call.
   ```bash
   modal deploy modal_inference.py
   ```

## Running the App with Modal

Once deployed, you can tell your local application to send video processing requests to Modal instead of trying to run them on your local CPU/GPU.

Simply set the `TRIBE_USE_MODAL` environment variable to `1` when starting the app:

```bash
# On Mac/Linux:
TRIBE_USE_MODAL=1 python app.py

# On Windows (PowerShell):
$env:TRIBE_USE_MODAL="1"; python app.py
```

## Advanced: H100 Batching (Experimental)

By default, the Modal deployment uses an **A100 GPU** with `float16` precision, `sdpa` (FlashAttention), and `TF32` enabled. This provides a highly stable encoding speed of ~1.2s to 1.4s per 64-frame clip.

If you want to maximize speed and don't mind experimental patches, you can switch to an **H100 GPU** and enable batched processing. This drops the encoding time to ~0.3s per clip (roughly 45 seconds total for a 1-minute video).

To enable this:
1. Open `modal_inference.py`.
2. Change `gpu="A100"` to `gpu="H100"` in the `@app.cls` decorator.
3. Change `USE_H100_BATCHING = False` to `USE_H100_BATCHING = True`.
4. Redeploy: `modal deploy modal_inference.py`

*⚠️ WARNING: Do not enable `USE_H100_BATCHING` on an A10G or A100 GPU. The memory spike will cause an Out of Memory (OOM) crash.*