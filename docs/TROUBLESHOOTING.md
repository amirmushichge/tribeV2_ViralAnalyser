# Troubleshooting

## The browser opens a JSON page instead of the app

Another local server may already be using port `8000`.

The launcher checks for the TRIBE app signature and will try ports `8001`
through `8010` if needed. Stop the other app or open the port printed in the
terminal window.

## PDF export fails

PDF export requires Google Chrome.

If Chrome is installed in a non-standard path, set:

```powershell
$env:TRIBE_CHROME_PATH = "C:\Path\To\chrome.exe"
```

On macOS, the app checks the standard Google Chrome, Microsoft Edge, and
Chromium app paths. If your browser is elsewhere, set:

```bash
export TRIBE_CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

## Hugging Face model download fails

Check that:

- you accepted any required model terms on Hugging Face;
- `huggingface-cli login` was run in the same operating-system user account;
- the token has access to the required model files;
- the machine has internet access.

## `Torch not compiled with CUDA enabled`

CUDA is optional. The app should automatically run TRIBE v2 on CPU when CUDA is not available.

If this error appears, update to the latest GitHub version and run `Start_TRIBE_Review.cmd` again. Newer builds force the TRIBE feature extractors to use the same verified device as the main model.

On macOS, CUDA is not available. Use CPU mode or opt into Apple Silicon MPS:

```bash
export TRIBE_DEVICE=cpu
export TRIBE_SPEECH_DEVICE=cpu
```

or:

```bash
export TRIBE_ENABLE_MPS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

MPS support is experimental. If a model path crashes or becomes unstable, force
CPU mode and retry.

## Video encoding is extremely slow

If a short clip shows a very long ETA during `Encoding video`, PyTorch is probably running without CUDA acceleration.

On Windows, run `Start_TRIBE_Review.cmd` again. The launcher checks for an NVIDIA GPU and installs CUDA-enabled PyTorch when needed. If CUDA still does not activate, update the NVIDIA driver and run the launcher again.

On macOS, expect CPU mode to be slower than CUDA. Test with short clips first.
If Apple Silicon MPS is enabled and performance is worse or unstable, force
CPU mode with `TRIBE_DEVICE=cpu`.

## `No space left on device`

CUDA-enabled PyTorch is large and can need several extra gigabytes while installing.

Free disk space on the drive where the app is extracted, then run the launcher
again. Keep at least `8 GB` free for the CUDA PyTorch install on Windows, and
more if the model cache and reports are stored on the same drive.

## Whisper or audio transcription fails

Check that FFmpeg is installed and available in `PATH`:

```powershell
ffmpeg -version
```

If transcription fails, the app should still continue with the visual/audio model outputs and a fallback speech state.

## TorchCodec / FFmpeg warnings

Some Python audio stacks warn when TorchCodec cannot load FFmpeg bindings. Install a compatible FFmpeg version and ensure it is visible in `PATH`.

The app also relies on FFmpeg through video/audio tooling, so fixing FFmpeg usually resolves this class of errors.

## The machine uses too much memory

Use shorter videos while testing and compare fewer variants at once. Comparison mode processes videos sequentially, but TRIBE and Whisper can still use significant memory.

For safest runs:

- test 1 video first;
- keep clips short;
- close other GPU/AI apps;
- prefer `TRIBE_CACHE_DIR` on a disk with enough space.

## Ollama is not installed

Ollama is optional. Without it, the app uses deterministic built-in wording for the recommendation layer.

## macOS launcher cannot find Python 3.11

Install Python 3.11 or newer:

```bash
brew install python@3.11
```

Then run:

```bash
./start_macos.sh
```

If you use a custom Python install, point the launcher at it:

```bash
export TRIBE_PYTHON=/path/to/python3.11
./start_macos.sh
```
