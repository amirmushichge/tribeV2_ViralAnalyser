# macOS Installation

This app can run locally on macOS with the official TRIBE v2 inference path.
The first supported macOS target is Apple Silicon with CPU execution. Apple
Silicon MPS acceleration is available as an opt-in experimental path.

## Requirements

Recommended local setup:

- macOS 14 or newer
- Apple Silicon M1 or newer
- Python 3.11 or newer
- Xcode Command Line Tools
- Google Chrome for PDF export
- 24 GB unified memory preferred
- 30 GB+ free disk space, preferably on SSD

The first launch downloads Python packages, official TRIBE v2 model files, the
Whisper speech model, and local video/audio tooling.

## 1. Install system tools

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Install Python 3.11 with Homebrew if you do not already have it:

```bash
brew install python@3.11
```

Google Chrome is recommended for PDF export. If Chrome is installed somewhere
non-standard, set `TRIBE_CHROME_PATH` before launching the app.

## 2. Download

Clone the repository or download the ZIP archive from GitHub, then open a
terminal in the extracted project folder.

Do not run the app directly from inside the ZIP preview.

## 3. First launch

Run:

```bash
./start_macos.sh
```

The first run creates `.venv`, installs Python dependencies, and downloads the
model files. This can take a while. Keep the terminal open until setup
finishes.

When the terminal says initial setup finished successfully, run the launcher
again:

```bash
./start_macos.sh
```

The browser should open automatically. If it does not, open the address printed
in the terminal, usually:

```url
http://127.0.0.1:8000
```

## 4. Optional Hugging Face login

If the model download fails because Hugging Face asks for access, run this
inside the project folder after `.venv` exists:

```bash
. .venv/bin/activate
huggingface-cli login
```

Paste a Hugging Face token that has access to the required model files, then
run `./start_macos.sh` again.

## 5. Optional cache path

By default, the app uses:

```text
~/Downloads/tribe_cache
```

To override it:

```bash
export TRIBE_CACHE_DIR="$HOME/tribe_cache"
```

## 6. Optional Apple Silicon MPS acceleration

CPU mode is the safest macOS default. To try Apple Silicon GPU acceleration:

```bash
export TRIBE_ENABLE_MPS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
./start_macos.sh
```

You can also explicitly request a device:

```bash
export TRIBE_DEVICE=mps
```

If MPS is unavailable or unsupported for a runtime path, the app falls back to
CPU. For troubleshooting, force CPU with:

```bash
export TRIBE_DEVICE=cpu
export TRIBE_SPEECH_DEVICE=cpu
```

## 7. Optional Ollama setup

The app works without Ollama. If Ollama is available, it can be used as a local
copy-rewriting layer.

Install Ollama, then pull one of the supported local models:

```bash
ollama pull qwen3.5:9b
```

If no supported Ollama model is found, the app falls back to deterministic
built-in copy.

## 8. Manual development commands

Create and activate the virtual environment manually:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python bootstrap_models.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Run a smoke test with a local video:

```bash
python smoke_test.py /path/to/test-video.mp4
```
