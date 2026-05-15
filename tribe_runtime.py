from __future__ import annotations

import pathlib
import json
import os
import re
import subprocess
import shutil
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import tribev2.eventstransforms as tribev2_eventstransforms

from runtime_setup import ensure_local_ffmpeg_on_path


if os.name == "nt" and hasattr(pathlib, "WindowsPath"):
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]

from tribev2 import TribeModel


ensure_local_ffmpeg_on_path()

MODEL_REPO = "facebook/tribev2"
DEFAULT_CACHE_DIR = Path.home() / "Downloads" / "tribe_cache"
CACHE_DIR = Path(os.environ.get("TRIBE_CACHE_DIR", DEFAULT_CACHE_DIR))
MODEL_SNAPSHOT_DIR = CACHE_DIR / "official_model_repo"
ENABLE_TEXT_EVENTS = os.environ.get("TRIBE_ENABLE_TEXT_EVENTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIRE_CUDA = os.environ.get("TRIBE_REQUIRE_CUDA", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
FORCE_CPU_EVENT_EXTRACTION = os.environ.get(
    "TRIBE_FORCE_CPU_EVENT_EXTRACTION",
    "",
).strip().lower() in {
    "1",
    "true",
    "yes",
}


@dataclass
class TribeRunResult:
    preds: Any
    timestamps: list[float]
    device: str
    modalities: list[str]
    events_count: int
    mesh_level: str
    subject_model: str
    hemodynamic_lag_seconds: float


class TribeVideoBackend:
    def __init__(self) -> None:
        self._model: TribeModel | None = None
        self._model_dir: Path | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return _select_torch_device()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def runtime_info(self) -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        gpu_name = None
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
        return {
            "device": self.device,
            "ready": self.ready,
            "requireCuda": REQUIRE_CUDA,
            "forceCpuEventExtraction": FORCE_CPU_EVENT_EXTRACTION,
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "torchCudaAvailable": cuda_available,
            "gpuName": gpu_name,
            "cacheDir": str(CACHE_DIR),
            "modelSnapshotDir": str(MODEL_SNAPSHOT_DIR),
        }

    def load(self) -> TribeModel:
        with self._lock:
            if self._model is None:
                device = self.device
                if REQUIRE_CUDA and device != "cuda":
                    raise RuntimeError("TRIBE_REQUIRE_CUDA is set, but CUDA is not available.")
                model_dir = self._resolve_official_model_dir(device)
                self._model = TribeModel.from_pretrained(
                    model_dir,
                    cache_folder=CACHE_DIR,
                    device=device,
                    config_update=_build_runtime_config_update(device),
                )
            return self._model

    def _resolve_official_model_dir(self, device: str) -> Path:
        if self._model_dir is not None:
            return self._model_dir

        from huggingface_hub import snapshot_download

        MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=["config.yaml", "best.ckpt"],
            local_dir=MODEL_SNAPSHOT_DIR,
            local_dir_use_symlinks=False,
        )
        self._model_dir = _prepare_runtime_model_dir(Path(snapshot_path), device)
        return self._model_dir

    def predict_video(self, video_path: str | Path) -> TribeRunResult:
        model = self.load()
        self._ensure_uvx_on_path()
        self._ensure_official_transcript_helper()
        device = self.device
        if REQUIRE_CUDA and device != "cuda":
            raise RuntimeError("TRIBE_REQUIRE_CUDA is set, but CUDA is not available.")
        events = self._get_events_dataframe(model, video_path, device)
        events = _drop_text_events_unless_enabled(events)
        if device == "cuda":
            torch.cuda.synchronize()
        predict_start = time.perf_counter()
        with torch.inference_mode():
            preds, segments = model.predict(events=events, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "event": "tribe_predict_completed",
                    "device": device,
                    "elapsedSeconds": round(time.perf_counter() - predict_start, 3),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        timestamps = [self._segment_timestamp(segment) for segment in segments]
        modalities = sorted({str(item).lower() for item in events["type"].unique().tolist()})
        return TribeRunResult(
            preds=preds,
            timestamps=timestamps,
            device=device,
            modalities=modalities,
            events_count=len(events),
            mesh_level="fsaverage5",
            subject_model="average",
            hemodynamic_lag_seconds=5.0,
        )

    def _get_events_dataframe(
        self,
        model: TribeModel,
        video_path: str | Path,
        device: str,
    ) -> pd.DataFrame:
        should_force_cpu = device != "cuda" or FORCE_CPU_EVENT_EXTRACTION
        if not should_force_cpu:
            return model.get_events_dataframe(video_path=str(Path(video_path)))

        original_cuda_check = tribev2_eventstransforms.torch.cuda.is_available
        tribev2_eventstransforms.torch.cuda.is_available = lambda: False
        try:
            return model.get_events_dataframe(video_path=str(Path(video_path)))
        finally:
            tribev2_eventstransforms.torch.cuda.is_available = original_cuda_check

    @staticmethod
    def _segment_timestamp(segment: Any) -> float:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        offset = float(getattr(segment, "offset", 0.0) or 0.0)
        return start + offset

    @staticmethod
    def _ensure_uvx_on_path() -> None:
        if shutil.which("uvx"):
            return
        candidates = [
            Path.home() / "Documents" / "ComfyUI" / ".venv" / "Scripts" / "uvx.exe",
            Path.home() / ".local" / "bin" / "uvx",
        ]
        for candidate in candidates:
            if candidate.exists():
                current_path = os.environ.get("PATH", "")
                os.environ["PATH"] = str(candidate.parent) + os.pathsep + current_path
                return

    @staticmethod
    def _ensure_official_transcript_helper() -> None:
        if not ENABLE_TEXT_EVENTS:
            current = tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio
            if getattr(current, "__name__", "") == "_skip_get_transcript_from_audio":
                return

            def _skip_get_transcript_from_audio(
                _wav_filename: Path,
                _language: str,
            ) -> pd.DataFrame:
                return _empty_transcript_dataframe()

            tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
                _skip_get_transcript_from_audio
            )
            return

        current = tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio
        if getattr(current, "__name__", "") == "_compatible_get_transcript_from_audio":
            return

        def _compatible_get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
            language_codes = dict(
                english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
            )
            if language not in language_codes:
                raise ValueError(f"Language {language} not supported")

            device = "cuda" if tribev2_eventstransforms.torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            batch_size = "4"

            with tempfile.TemporaryDirectory() as output_dir:
                cmd = [
                    "uvx",
                    "whisperx",
                    str(wav_filename),
                    "--model",
                    "large-v3",
                    "--language",
                    language_codes[language],
                    "--device",
                    device,
                    "--compute_type",
                    compute_type,
                    "--batch_size",
                    batch_size,
                    "--align_model",
                    "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                    "--output_dir",
                    output_dir,
                    "--output_format",
                    "json",
                ]
                cmd = [c for c in cmd if c]
                env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
                env.setdefault("PYTHONIOENCODING", "utf-8")
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                    )
                except OSError:
                    return _empty_transcript_dataframe()
                if result.returncode != 0:
                    return _empty_transcript_dataframe()

                json_path = Path(output_dir) / f"{wav_filename.stem}.json"
                if not json_path.exists():
                    return _empty_transcript_dataframe()
                transcript = json.loads(json_path.read_text(encoding="utf-8"))

            words = []
            for i, segment in enumerate(transcript.get("segments") or []):
                sentence = segment["text"].replace('"', "")
                for word in segment.get("words", []):
                    if "start" not in word:
                        continue
                    words.append(
                        {
                            "text": word["word"].replace('"', ""),
                            "start": word["start"],
                            "duration": word["end"] - word["start"],
                            "sequence_id": i,
                            "sentence": sentence,
                        }
                    )
            return pd.DataFrame(words)

        tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
            _compatible_get_transcript_from_audio
        )


def _empty_transcript_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["text", "start", "duration", "sequence_id", "sentence"])


def _drop_text_events_unless_enabled(events: pd.DataFrame) -> pd.DataFrame:
    if ENABLE_TEXT_EVENTS or "type" not in events.columns:
        return events
    # The app keeps its own local Whisper speech layer. Official TRIBE Word events
    # require the gated LLaMA text encoder, so skip them by default for clean installs.
    event_type = events["type"].astype(str).str.lower()
    return events[event_type != "word"].reset_index(drop=True)


def _select_torch_device() -> str:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return "cpu"
    try:
        torch.empty(1).to("cuda")
    except Exception:
        return "cpu"
    return "cuda"


def _build_runtime_config_update(device: str) -> dict[str, Any]:
    image_batch_size = 1
    video_batch_size = 1
    return {
        "data.num_workers": 0,
        "data.batch_size": 1,
        "data.text_feature.device": device,
        "data.audio_feature.device": device,
        "data.image_feature.image.device": device,
        "data.video_feature.image.device": device,
        "data.image_feature.image.batch_size": image_batch_size,
        "data.video_feature.image.batch_size": video_batch_size,
    }


def _prepare_runtime_model_dir(snapshot_dir: Path, device: str) -> Path:
    source_config = snapshot_dir / "config.yaml"
    source_checkpoint = snapshot_dir / "best.ckpt"
    if not source_config.exists() or not source_checkpoint.exists():
        return snapshot_dir

    runtime_dir = snapshot_dir / f"runtime-{device}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = runtime_dir / "config.yaml"
    config_text = source_config.read_text(encoding="utf-8")
    config_text = re.sub(
        r"(?m)^(\s*device:\s*)cuda\s*$",
        lambda match: f"{match.group(1)}{device}",
        config_text,
    )
    runtime_config.write_text(config_text, encoding="utf-8")

    runtime_checkpoint = runtime_dir / "best.ckpt"
    if not runtime_checkpoint.exists():
        try:
            os.link(source_checkpoint, runtime_checkpoint)
        except OSError:
            runtime_checkpoint.write_bytes(source_checkpoint.read_bytes())

    return runtime_dir
