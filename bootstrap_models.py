from __future__ import annotations

import json
import os
import time
from pathlib import Path


BOOTSTRAP_DIR = Path(".bootstrap")
READY_FILE = BOOTSTRAP_DIR / "models-ready.json"


def _log(message: str) -> None:
    print(message, flush=True)


def _write_ready(cache_dir: Path) -> None:
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    READY_FILE.write_text(
        json.dumps(
            {
                "ready": True,
                "cache_dir": str(cache_dir),
                "created_at": int(time.time()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _is_ready() -> bool:
    if not READY_FILE.exists():
        return False
    try:
        data = json.loads(READY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("ready"))


def _launcher_name() -> str:
    return "Start_TRIBE_Review.cmd" if os.name == "nt" else "./start_macos.sh"


def main() -> int:
    if _is_ready():
        _log("Model bootstrap already completed. Starting the app now.")
        return 0

    _log("")
    _log("First launch setup")
    _log("The app is preparing everything needed for TRIBE v2 analysis.")
    _log("This can take a while on the first run because Python packages and model files are downloaded now.")
    _log("Please keep this terminal open. Later launches will be much faster.")
    _log("")

    try:
        from speech_runtime import SpeechTranscriber, WHISPER_CACHE_DIR
        from tribe_runtime import CACHE_DIR, MODEL_REPO, MODEL_SNAPSHOT_DIR, TribeVideoBackend
    except Exception as exc:
        _log("Could not import the app runtime after installing dependencies.")
        _log(f"Error: {exc}")
        return 1

    try:
        _log(f"Step 1/3: downloading TRIBE v2 model files from Hugging Face ({MODEL_REPO}).")
        from huggingface_hub import snapshot_download

        MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=["config.yaml", "best.ckpt"],
            local_dir=MODEL_SNAPSHOT_DIR,
            local_dir_use_symlinks=False,
        )

        _log("Step 2/3: loading the TRIBE v2 model once to verify the checkpoint.")
        TribeVideoBackend().load()

        _log(f"Step 3/3: downloading the Whisper speech model into {WHISPER_CACHE_DIR}.")
        SpeechTranscriber().load()

        _write_ready(CACHE_DIR)
        _log("")
        _log("Setup complete.")
        _log(f"Close this terminal, then start the app again with {_launcher_name()}.")
        _log("After this first setup, launches and analyses should start much faster.")
        return 0
    except Exception as exc:
        _log("")
        _log("First launch setup failed.")
        _log(f"Error: {exc}")
        _log("")
        _log("Check your internet connection and make sure Hugging Face access is available if the model download asks for it.")
        _log(f"After fixing the issue, run {_launcher_name()} again.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
