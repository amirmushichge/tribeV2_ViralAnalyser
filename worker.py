from __future__ import annotations

import json
import math
import os
import shutil
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from google.cloud import storage

from brain_visualization import REGION_DEFINITIONS, build_brain_simulation
from official_report import generate_official_report
from tribe_runtime import TribeVideoBackend


app = FastAPI(title="Worthy TRIBE Worker")
backend = TribeVideoBackend()

MAX_HEADLINE_LENGTH = 240
DOWNLOAD_TIMEOUT_SECONDS = 300
REPORT_JSON_NAME = "report.json"
BRAIN_TIMELINE_JSON_NAME = "brain.json"
BRAIN_POSTER_NAME = "brain-poster.png"


class RequestError(ValueError):
    pass


class WorkerConfigError(RuntimeError):
    pass


@app.post("/scan")
async def scan_video(request: FastAPIRequest) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return _failed("request_invalid", "Request body must be JSON.", False, 400)

    try:
        result = _run_scan(payload)
    except RequestError as exc:
        return _failed("request_invalid", str(exc), False, 400)
    except WorkerConfigError as exc:
        return _failed("worker_config_invalid", str(exc), False, 500)
    except Exception as exc:
        return _failed("worker_exception", str(exc), True, 500)

    return JSONResponse(result)


def _run_scan(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError("Request body must be an object.")

    scan_id = _required_string(payload, "scanId")
    partner_id = _required_string(payload, "partnerId")
    input_url = _required_string(payload, "inputUrl")
    output_prefix = _required_string(payload, "outputPrefix").strip("/")
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    variant_name = _variant_name(source, scan_id)

    expected_prefix = f"tribe-video-scans/{partner_id}/"
    if not output_prefix.startswith(expected_prefix):
        raise RequestError(f"outputPrefix must start with {expected_prefix}.")

    with TemporaryDirectory(prefix=f"tribe-{scan_id}-") as tmp_dir:
        video_path = Path(tmp_dir) / "input.mp4"
        _download(input_url, video_path)

        run = backend.predict_video(video_path)
        report = generate_official_report(video_path, run, variant_name=variant_name)
        brain_timeline = _build_brain_timeline(run)
        brain_timeline_path = f"{output_prefix}/{BRAIN_TIMELINE_JSON_NAME}"
        brain_poster_path = f"{output_prefix}/{BRAIN_POSTER_NAME}"
        raw_report_path = f"{output_prefix}/{REPORT_JSON_NAME}"
        _upload_json(brain_timeline_path, brain_timeline)
        _upload_bytes(brain_poster_path, _render_brain_poster(brain_timeline), "image/png")
        _upload_json(raw_report_path, report)

    summary = _build_summary(report, brain_timeline_path, brain_poster_path)
    return {
        "status": "completed",
        "summary": summary,
        "rawReportPath": raw_report_path,
    }


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{key} is required.")
    return value.strip()


def _variant_name(source: dict[str, Any], scan_id: str) -> str:
    concept_name = source.get("conceptName")
    if isinstance(concept_name, str) and concept_name.strip():
        return concept_name.strip()
    return scan_id


def _download(input_url: str, video_path: Path) -> None:
    request = Request(input_url, headers={"User-Agent": "worthy-tribe-worker/1.0"})
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with video_path.open("wb") as output:
            shutil.copyfileobj(response, output)


def _upload_json(path: str, payload: dict[str, Any]) -> None:
    bucket_name = _bucket_name()
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    storage.Client().bucket(bucket_name).blob(path).upload_from_string(
        body,
        content_type="application/json",
    )


def _upload_bytes(path: str, payload: bytes, content_type: str) -> None:
    bucket_name = _bucket_name()
    storage.Client().bucket(bucket_name).blob(path).upload_from_string(
        payload,
        content_type=content_type,
    )


def _bucket_name() -> str:
    for key in ("FIREBASE_STORAGE_BUCKET", "GCS_BUCKET"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise WorkerConfigError("Set FIREBASE_STORAGE_BUCKET or GCS_BUCKET.")


def _build_summary(
    report: dict[str, Any],
    brain_timeline_path: str,
    brain_poster_path: str,
) -> dict[str, Any]:
    score = _clamp_score(_read_number(report, ["timeline", "avg_score"], None))
    peak_time = _read_number(report, ["predictions", "peak_time_seconds"], None)
    headline = _headline(report)

    summary: dict[str, Any] = {
        "band": _band(score) if score is not None else "mixed",
        "brainTimelinePath": brain_timeline_path,
        "brainPosterPath": brain_poster_path,
    }
    if score is not None:
        summary["score"] = score
    if peak_time is not None and math.isfinite(peak_time) and peak_time >= 0:
        summary["peakTimeSeconds"] = round(float(peak_time), 2)
    if headline:
        summary["headline"] = headline[:MAX_HEADLINE_LENGTH]
    return summary


def _build_brain_timeline(run: Any) -> dict[str, Any]:
    simulation = build_brain_simulation(run.preds, run.timestamps)
    frames = simulation.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("TRIBE brain simulation produced no frames.")

    timestamps: list[float] = []
    regions: dict[str, list[float]] = {region["key"]: [] for region in REGION_DEFINITIONS}

    for frame in frames:
        if not isinstance(frame, dict):
            raise RuntimeError("TRIBE brain simulation frame is invalid.")

        raw_second = frame.get("seconds")
        second = float(raw_second) if isinstance(raw_second, (int, float)) else 0.0
        if timestamps and second <= timestamps[-1]:
            second = timestamps[-1] + 0.001
        timestamps.append(round(second, 3))

        region_scores = frame.get("region_scores")
        if not isinstance(region_scores, list):
            raise RuntimeError("TRIBE brain simulation frame is missing region scores.")

        for index, region in enumerate(REGION_DEFINITIONS):
            score = region_scores[index] if index < len(region_scores) else 0.0
            regions[region["key"]].append(_clamp_unit(score))

    return {
        "version": 1,
        "durationSeconds": max(timestamps[-1], 0.001),
        "timestamps": timestamps,
        "regions": regions,
    }


def _render_brain_poster(timeline: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = timeline["timestamps"]
    regions = timeline["regions"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    fig.patch.set_facecolor("#06131f")
    ax.set_facecolor("#06131f")

    for region in REGION_DEFINITIONS:
        values = regions[region["key"]]
        ax.plot(
            timestamps,
            values,
            color=region["color"],
            linewidth=2.0,
            alpha=0.86,
            label=region["label_en"],
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(timestamps[-1], 0.001))
    ax.grid(color="#244255", linewidth=0.6, alpha=0.35)
    ax.tick_params(colors="#8bb8c7", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a4d60")
    ax.set_title("TRIBE attention timeline", color="#e6fbff", fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor="#d7eef5")
    fig.tight_layout(pad=1.0)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def _read_number(report: dict[str, Any], path: list[str], fallback: float | None) -> float | None:
    current: Any = report
    for key in path:
        if not isinstance(current, dict):
            return fallback
        current = current.get(key)
    if isinstance(current, (int, float)) and math.isfinite(float(current)):
        return float(current)
    return fallback


def _clamp_score(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(max(0.0, min(100.0, float(value))), 1)


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _band(score: float) -> str:
    if score >= 65:
        return "strong"
    if score < 45:
        return "weak"
    return "mixed"


def _headline(report: dict[str, Any]) -> str:
    simple_readout = report.get("simple_readout")
    if isinstance(simple_readout, dict):
        english = simple_readout.get("en")
        if isinstance(english, dict):
            summary_body = english.get("summary_body")
            if isinstance(summary_body, str) and summary_body.strip():
                return summary_body.strip()
    return "TRIBE scan completed."


def _failed(code: str, message: str, retryable: bool, status_code: int) -> JSONResponse:
    return JSONResponse(
        {
            "status": "failed",
            "error": {
                "code": code[:80],
                "message": message[:500],
                "retryable": retryable,
            },
        },
        status_code=status_code,
    )
