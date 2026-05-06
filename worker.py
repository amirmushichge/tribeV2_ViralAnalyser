from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from google.cloud import storage

from official_report import generate_official_report
from tribe_runtime import TribeVideoBackend


app = FastAPI(title="Worthy TRIBE Worker")
backend = TribeVideoBackend()

MAX_HEADLINE_LENGTH = 240
DOWNLOAD_TIMEOUT_SECONDS = 300
REPORT_JSON_NAME = "report.json"


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
        raw_report_path = f"{output_prefix}/{REPORT_JSON_NAME}"
        _upload_report(raw_report_path, report)

    summary = _build_summary(report)
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


def _upload_report(raw_report_path: str, report: dict[str, Any]) -> None:
    bucket_name = _bucket_name()
    body = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    storage.Client().bucket(bucket_name).blob(raw_report_path).upload_from_string(
        body,
        content_type="application/json",
    )


def _bucket_name() -> str:
    for key in ("FIREBASE_STORAGE_BUCKET", "GCS_BUCKET"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise WorkerConfigError("Set FIREBASE_STORAGE_BUCKET or GCS_BUCKET.")


def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
    score = _clamp_score(_read_number(report, ["timeline", "avg_score"], 50.0))
    peak_time = _read_number(report, ["predictions", "peak_time_seconds"], None)
    headline = _headline(report)

    summary: dict[str, Any] = {
        "band": _band(score),
        "score": score,
    }
    if peak_time is not None and math.isfinite(peak_time) and peak_time >= 0:
        summary["peakTimeSeconds"] = round(float(peak_time), 2)
    if headline:
        summary["headline"] = headline[:MAX_HEADLINE_LENGTH]
    return summary


def _read_number(report: dict[str, Any], path: list[str], fallback: float | None) -> float | None:
    current: Any = report
    for key in path:
        if not isinstance(current, dict):
            return fallback
        current = current.get(key)
    if isinstance(current, (int, float)) and math.isfinite(float(current)):
        return float(current)
    return fallback


def _clamp_score(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        value = 50.0
    return round(max(0.0, min(100.0, float(value))), 1)


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
