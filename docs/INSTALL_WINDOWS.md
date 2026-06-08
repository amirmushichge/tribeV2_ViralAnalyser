# Windows Installation

This app is designed as a local Windows workflow for reviewing videos with Meta TRIBE v2.

## Requirements

Recommended local setup:

- Windows 10/11 64-bit
- Python 3.11
- 16 GB RAM
- Modern 8-core CPU or better
- NVIDIA GPU
- 6 GB VRAM minimum, 12 GB+ preferred
- 30 GB+ free disk space, preferably on SSD

The first launch downloads Python packages, official TRIBE v2 model files, the Whisper speech model, and local video/audio tooling.

Website URL review requires Google Chrome or Microsoft Edge. The app uses the local browser only to capture screenshots of the page.

Visual grid review works with normal image files such as PNG, JPG, JPEG, or WEBP.

## 1. Download

1. Open the GitHub repository page.
2. Click the green `Code` button.
3. Click `Download ZIP`.
4. Wait for the ZIP file to finish downloading.

## 2. Extract

1. Right-click the downloaded ZIP file.
2. Click `Extract All`.
3. Choose a normal folder such as `Desktop` or `Downloads`.
4. Open the extracted folder.

Do not run the app directly from inside the ZIP preview.

## 3. First launch

1. Double-click `Start_TRIBE_Review.cmd`.
2. A black terminal window will open.
3. Keep it open. The app is creating its local environment and downloading the model files.
4. The first setup can take a while.
5. When the terminal says setup is complete, close the terminal.

## 4. Start the app

Double-click `Start_TRIBE_Review.cmd` again.

The browser should open automatically. If it does not, open this address manually:

```url
http://127.0.0.1:8000
```

Later launches are much faster because the setup files are already installed.

## Website URL review

Paste a landing-page URL into the Website review form.

The app captures:

- desktop full-page screenshot
- mobile full-page screenshot
- original and heatmap views
- fold-by-fold section scores
- optional before/after comparison when a second URL is provided

This is not eye-tracking. It is a visual attention estimate for layout review.

## Visual grid review

Upload one image that contains several visual options in a grid.

The app shows:

- original and heatmap views
- top-3 visual picks
- per-cell attention scores
- short reasons for why each option is stronger or weaker

This is a fast attention pre-check for static creatives, not a guaranteed performance prediction.

## 5. Optional Hugging Face login

If the model download fails because Hugging Face asks for access, run this inside the project folder after `.venv` exists:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

Paste a Hugging Face token that has access to the required model files, then run `Start_TRIBE_Review.cmd` again.

## 6. Optional cache path

By default, the app uses:

```text
%USERPROFILE%\Downloads\tribe_cache
```

To override it:

```powershell
$env:TRIBE_CACHE_DIR = "D:\tribe_cache"
```

## 7. Optional Ollama setup

The app works without Ollama. If Ollama is available, it can be used as a local copy-rewriting layer.

Install Ollama, then pull one of the supported local models:

```powershell
ollama pull qwen3.5:9b
```

If no supported Ollama model is found, the app falls back to deterministic built-in copy.
