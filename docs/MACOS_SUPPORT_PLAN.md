# macOS Support Plan

This document outlines the work required to make TRIBE Review MVP run on
macOS in addition to the current Windows-first workflow, and records the
implementation status for the macOS support branch.

The expected first target is reliable local execution on Apple Silicon using
CPU fallback, with optional experimental MPS acceleration. CUDA-class
performance should not be assumed on macOS.

## Goals

- Support local setup and launch on macOS 14+ Apple Silicon.
- Keep the current Windows launcher and behavior intact.
- Provide a clean CPU path for machines without CUDA.
- Add an opt-in MPS path for Apple Silicon validation.
- Document expected performance and known limitations.
- Keep runtime media, model caches, and reports out of Git.

## Non-goals

- Replacing the official TRIBE v2 inference path.
- Guaranteeing NVIDIA CUDA-level performance on Apple Silicon.
- Supporting Intel macOS as a primary accelerated target.
- Training TRIBE v2 locally on macOS.
- Bundling model weights, Whisper weights, or Hugging Face credentials.

## Implementation Status

Completed in this branch:

- Added `start_macos.sh`.
- Added `docs/INSTALL_MACOS.md`.
- Updated README and troubleshooting docs.
- Added macOS Chrome, Edge, and Chromium detection.
- Guarded the Windows-only pathlib compatibility patch.
- Added `TRIBE_DEVICE`, `TRIBE_SPEECH_DEVICE`, and opt-in `TRIBE_ENABLE_MPS`
  handling.
- Updated bootstrap messages to point at the platform launcher.

Remaining validation:

- Full dependency installation on a clean Apple Silicon machine.
- First-run Hugging Face model bootstrap on macOS.
- Short-video CPU smoke test.
- Short-video MPS opt-in smoke test.
- PDF export through the browser UI.

## Original macOS Blockers

1. `tribe_runtime.py` applied a pathlib monkeypatch on all platforms:

   ```python
   if hasattr(pathlib, "WindowsPath"):
       pathlib.PosixPath = pathlib.WindowsPath
   ```

   On macOS, `WindowsPath` exists as a class but cannot be instantiated. This
   can break normal `Path(...)` construction. This branch guards the patch so
   it only runs on Windows.

2. The launcher is Windows-specific:

   - `Start_TRIBE_Review.cmd`
   - `start_mvp.ps1`
   - `.venv\Scripts\python.exe`
   - PowerShell browser opening
   - CUDA PyTorch installation logic

3. PDF export only searched Windows browser paths in `pdf_report.py`.

4. Device selection only supported CUDA or CPU:

   - `tribe_runtime.py`
   - `speech_runtime.py`

5. Installation docs and troubleshooting were Windows-centered.

6. The app has not been validated against Apple Silicon dependency wheels for
   TRIBE v2, PyTorch, openai-whisper, MoviePy, FFmpeg, and plotting extras.

## Proposed Implementation Phases

### Phase 1: macOS-safe runtime fixes

- Guard the pathlib compatibility patch with an operating-system check.
- Confirm all project paths use `pathlib.Path` or platform-neutral APIs.
- Keep `runtime_media/`, `.bootstrap/`, and model caches relative or
  user-configurable.
- Add a small runtime helper for platform detection if repeated checks appear.

Suggested acceptance checks:

```bash
python -m py_compile app.py bootstrap_models.py tribe_runtime.py speech_runtime.py official_report.py review_engine.py report_localization.py pdf_report.py brain_visualization.py runtime_setup.py
python -c "from pathlib import Path; import tribe_runtime; print(Path('ok'))"
```

### Phase 2: Device selection for Apple Silicon

- Add a device selection helper that supports:

  1. explicit `TRIBE_DEVICE`
  2. CUDA when available
  3. MPS when enabled and available
  4. CPU fallback

- Recommended environment variables:

  ```bash
  TRIBE_DEVICE=auto
  TRIBE_ENABLE_MPS=0
  PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

- Keep MPS opt-in initially. A conservative policy is:

  - default: `cuda -> cpu`
  - opt-in Apple Silicon: `cuda -> mps -> cpu`

- Update `tribe_runtime.py` to pass the selected device into
  `TribeModel.from_pretrained(...)` and runtime config overrides.
- Update `speech_runtime.py` separately. Whisper may work on MPS, but should
  be allowed to use CPU independently if MPS is unstable.
- Add logging or report metadata showing the selected device.

Suggested acceptance checks:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
TRIBE_ENABLE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -c "from tribe_runtime import TribeVideoBackend; print(TribeVideoBackend().device)"
```

### Phase 3: macOS launcher

- Add `start_macos.sh`.
- Use Python 3.11+ from `python3.11`, Homebrew, or a user-provided
  interpreter.
- Create `.venv` with POSIX paths:

  ```bash
  python3.11 -m venv .venv
  . .venv/bin/activate
  ```

- Install dependencies with:

  ```bash
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
  ```

- Run `bootstrap_models.py` once when `.bootstrap/models-ready.json` does not
  exist.
- Select the first available port from `8000` through `8010`.
- Open the browser with `open "$url"` after the server responds.
- Print clear messages for missing Python 3.11, dependency install failure,
  Hugging Face access failure, and model bootstrap failure.

Suggested acceptance checks:

```bash
chmod +x start_macos.sh
./start_macos.sh --no-browser
curl -I http://127.0.0.1:8000
```

### Phase 4: PDF export on macOS

- Extend `_find_chrome_executable()` in `pdf_report.py` to check:

  ```text
  /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
  /Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge
  /Applications/Chromium.app/Contents/MacOS/Chromium
  ```

- Preserve `TRIBE_CHROME_PATH` as the highest-priority override.
- Add docs for setting `TRIBE_CHROME_PATH` when Chrome is installed elsewhere.

Suggested acceptance checks:

```bash
python -c "from pdf_report import _find_chrome_executable; print(_find_chrome_executable())"
```

### Phase 5: macOS installation docs

- Add `docs/INSTALL_MACOS.md`.
- Cover:

  - Apple Silicon recommendation.
  - Python 3.11+ installation.
  - Xcode Command Line Tools.
  - FFmpeg expectations.
  - Hugging Face login.
  - `TRIBE_CACHE_DIR`.
  - `TRIBE_CHROME_PATH`.
  - CPU versus MPS expectations.
  - first-run model download time.
  - memory and disk requirements.

- Update `README.md` with a short macOS section and link to the new guide.
- Update `docs/TROUBLESHOOTING.md` with macOS-specific notes.

Suggested macOS baseline:

- macOS 14+
- Apple Silicon M1 or newer
- Python 3.11
- 24 GB unified memory preferred
- 30 GB+ free disk space
- Google Chrome for PDF export

### Phase 6: Validation matrix

Run the following matrix before marking macOS support stable:

| Platform | Device mode | Expected result |
| --- | --- | --- |
| Windows 10/11 NVIDIA | CUDA | Existing launcher still works |
| Windows 10/11 no CUDA | CPU | App starts and analyzes short video |
| macOS Apple Silicon | CPU | App starts and analyzes short video |
| macOS Apple Silicon | MPS opt-in | App starts and either analyzes or falls back cleanly |
| macOS Apple Silicon | PDF export | PDF downloads successfully when Chrome is installed |

Minimum functional tests:

```bash
python -m py_compile app.py bootstrap_models.py tribe_runtime.py speech_runtime.py official_report.py review_engine.py report_localization.py pdf_report.py brain_visualization.py runtime_setup.py
python smoke_test.py /path/to/short-test-video.mp4
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Manual UI tests:

- Open the root page.
- Upload one short `.mp4`.
- Confirm analysis completes.
- Confirm video playback works.
- Confirm JSON report download works.
- Confirm PDF report download works.
- Compare two short videos.
- Restart the app and open a saved report.

### Phase 7: CI and packaging follow-up

If the project adopts CI, add lightweight checks first:

- Syntax check on macOS and Windows.
- Import smoke test without model downloads.
- Unit test for device selection.
- Unit test for Chrome path detection.
- ShellCheck for `start_macos.sh`, if available.

Avoid full model download and inference in default CI because the model files
are large and may require Hugging Face access.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| MPS unsupported operation | Keep MPS opt-in and document `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| MPS memory pressure | Default to CPU until validated; recommend short clips first |
| Whisper instability on MPS | Allow speech runtime to force CPU independently |
| TRIBE dependency wheel unavailable | Pin or document compatible Python/PyTorch versions |
| PDF export path mismatch | Keep `TRIBE_CHROME_PATH` override |
| Setup time appears frozen | Add clear launcher progress messages |
| Hugging Face access fails | Document `huggingface-cli login` inside `.venv` |

## Proposed PR Breakdown

To keep review manageable, split implementation into small PRs:

1. Add macOS support plan and docs skeleton.
2. Fix platform-safe path handling.
3. Add macOS launcher.
4. Add macOS Chrome detection.
5. Add explicit device selection and MPS opt-in.
6. Add validation results from a real Apple Silicon smoke test.

## Definition of Done

macOS support should be considered complete when:

- `./start_macos.sh` can create `.venv`, install dependencies, bootstrap
  models, and start the app on Apple Silicon.
- A short video can be analyzed on CPU.
- MPS opt-in either accelerates successfully or falls back to CPU without
  crashing.
- PDF export works with Chrome installed.
- Windows startup remains unchanged.
- README and troubleshooting docs explain the macOS path clearly.
