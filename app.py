from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

from analysis_settings import DEFAULT_ANALYSIS_MODE
from brain_visualization import build_brain_simulation
from ollama_runtime import simplify_review_copy
from official_report import generate_official_report
from pdf_report import render_html_pdf
from review_engine_runtime_patch import apply as apply_review_engine_patch
from report_localization import (
    get_ui_texts,
    localize_report,
    normalize_report_language,
)
from review_engine import generate_comparison_report, generate_review
from speech_runtime import SpeechTranscriber
from tribe_runtime import TribeVideoBackend
from visual_grid_analysis import build_visual_grid_report
from website_analysis import build_website_attention_report
from website_capture import capture_website_screenshot, get_viewport_config


apply_review_engine_patch()

APP_DIR = Path(__file__).resolve().parent
MEDIA_DIR = APP_DIR / "runtime_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app = FastAPI(title="TRIBE Review MVP")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
backend = TribeVideoBackend()
speech_backend = SpeechTranscriber()
REPORTS: OrderedDict[str, dict] = OrderedDict()
MAX_REPORTS = 24
REPORT_JSON_NAME = "report.json"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, lang: str | None = None) -> HTMLResponse:
    language = normalize_report_language(lang)
    return _render_page(
        request,
        result=None,
        error=None,
        language=language,
    )


@app.get("/reports/{report_id}.json")
async def get_report(report_id: str, lang: str | None = None) -> JSONResponse:
    report = _get_stored_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    localized_report = _get_localized_report(report, normalize_report_language(lang))
    return JSONResponse(_public_report(localized_report))


@app.get("/reports/{report_id}.pdf")
async def get_report_pdf(report_id: str, lang: str | None = None) -> StreamingResponse:
    report = _get_stored_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    language = normalize_report_language(lang)
    localized_report = _get_localized_report(report, language)
    try:
        pdf_bytes = render_html_pdf(_render_pdf_html(localized_report, language))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    filename = f"tribe-report-{report_id}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/media/{report_id}/{variant_key}")
async def get_media(report_id: str, variant_key: str) -> FileResponse:
    report = _get_stored_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    media_path = report.get("_media_path") if variant_key == "v1" else None
    media_path = media_path or _find_media_file(report_id, variant_key)
    if media_path:
        return FileResponse(media_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/media/{report_id}/website/{asset_name}")
async def get_website_asset(report_id: str, asset_name: str) -> FileResponse:
    if not re.match(r"^[a-z0-9_-]+\.(?:png|jpg|jpeg)$", asset_name, re.IGNORECASE):
        raise HTTPException(status_code=404, detail="Asset not found")
    asset_path = MEDIA_DIR / report_id / asset_name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = "image/png" if asset_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(asset_path, media_type=media_type)


@app.get("/media/{report_id}/assets/{asset_name}")
async def get_image_asset(report_id: str, asset_name: str) -> FileResponse:
    if not re.match(r"^[a-z0-9_-]+\.(?:png|jpg|jpeg)$", asset_name, re.IGNORECASE):
        raise HTTPException(status_code=404, detail="Asset not found")
    asset_path = MEDIA_DIR / report_id / asset_name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = "image/png" if asset_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(asset_path, media_type=media_type)


@app.get("/reports/{report_id}", response_class=HTMLResponse)
async def view_report(request: Request, report_id: str, lang: str | None = None) -> HTMLResponse:
    report = _get_stored_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    language = normalize_report_language(lang)
    return _render_page(
        request,
        result=_get_localized_report(report, language),
        error=None,
        language=language,
    )


@app.post("/review", response_class=HTMLResponse)
async def review_video(
    request: Request,
    videos: list[UploadFile] = File(...),
    analysis_mode: str = Form(DEFAULT_ANALYSIS_MODE),
    lang: str | None = None,
) -> HTMLResponse:
    report_id = uuid4().hex[:12]
    report_media_dir = MEDIA_DIR / report_id
    language = normalize_report_language(lang)
    selected_analysis_mode = _normalize_analysis_mode(analysis_mode)

    try:
        valid_uploads = [video for video in videos if video.filename]
        if not valid_uploads:
            raise ValueError("Upload one to four video files to run the TRIBE v2 workflow.")
        if len(valid_uploads) > 4:
            raise ValueError("Comparison mode supports up to 4 videos at once.")

        analyzed_variants: list[dict[str, Any]] = []
        editorial_variants: list[dict[str, Any]] = []
        for index, upload in enumerate(valid_uploads, start=1):
            variant_key = f"v{index}"
            result, editorial = await _analyze_upload(
                upload=upload,
                report_id=report_id,
                report_media_dir=report_media_dir,
                variant_key=variant_key,
                analysis_mode=selected_analysis_mode,
            )
            analyzed_variants.append(result)
            if editorial:
                editorial_variant = deepcopy(editorial)
                editorial_variant["variant_key"] = variant_key
                editorial_variant["title"] = result.get("title") or result.get("variant_name") or variant_key
                editorial_variant["variant_name"] = result.get("variant_name") or editorial_variant["title"]
                editorial_variant["media_url"] = result.get("media_url")
                editorial_variant["video"] = deepcopy(result.get("video"))
                editorial_variant["timeline"] = deepcopy(result.get("timeline"))
                editorial_variant["predictions"] = deepcopy(result.get("predictions"))
                editorial_variant["brain_simulation"] = deepcopy(result.get("brain_simulation"))
                editorial_variants.append(editorial_variant)

        if len(analyzed_variants) == 1:
            result = analyzed_variants[0]
        else:
            if len(editorial_variants) != len(analyzed_variants):
                raise ValueError("Comparison needs the local recommendation layer for every uploaded video.")
            result = generate_comparison_report(editorial_variants, analysis_mode=selected_analysis_mode)
            result["report_id"] = report_id
            result["created_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            result["timeline_overlay"] = _build_timeline_overlay(result.get("variants", []))
            result["official_sources"] = analyzed_variants[0].get("official_sources", {})

        _store_report(report_id, result)
        return _render_page(
            request,
            result=_get_localized_report(result, language),
            error=None,
            language=language,
        )
    except Exception as exc:
        return _render_page(
            request,
            result=None,
            error=_format_error(exc, language),
            language=language,
            status_code=500,
        )


@app.post("/review-url", response_class=HTMLResponse)
async def review_website_url(
    request: Request,
    website_url: str = Form(...),
    compare_website_url: str | None = Form(None),
    analysis_mode: str = Form(DEFAULT_ANALYSIS_MODE),
    lang: str | None = None,
) -> HTMLResponse:
    report_id = uuid4().hex[:12]
    report_media_dir = MEDIA_DIR / report_id
    language = normalize_report_language(lang)
    selected_analysis_mode = _normalize_analysis_mode(analysis_mode)

    try:
        urls = [website_url.strip()]
        if compare_website_url and compare_website_url.strip():
            urls.append(compare_website_url.strip())
        if len(urls) > 2:
            raise ValueError("Website comparison supports up to 2 URLs.")

        variants = []
        for index, url in enumerate(urls, start=1):
            variants.append(await _build_website_variant(url, report_id, report_media_dir, index))

        result = {
            "mode": "website",
            "source_kind": "website",
            "report_id": report_id,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "title": _website_variant_name(website_url),
            "website_url": website_url.strip(),
            "website": {
                "variants": variants,
                "comparison": _build_website_comparison(variants),
            },
            "analysis_mode": {"key": selected_analysis_mode, "label": selected_analysis_mode},
        }
        _store_report(report_id, result)
        return _render_page(
            request,
            result=_get_localized_report(result, language),
            error=None,
            language=language,
        )
    except Exception as exc:
        return _render_page(
            request,
            result=None,
            error=_format_error(exc, language),
            language=language,
            status_code=500,
        )


@app.post("/review-visual-grid", response_class=HTMLResponse)
async def review_visual_grid(
    request: Request,
    grid_image: UploadFile = File(...),
    grid_columns: int | None = Form(None),
    grid_rows: int | None = Form(None),
    analysis_mode: str = Form(DEFAULT_ANALYSIS_MODE),
    lang: str | None = None,
) -> HTMLResponse:
    report_id = uuid4().hex[:12]
    report_media_dir = MEDIA_DIR / report_id
    language = normalize_report_language(lang)
    selected_analysis_mode = _normalize_analysis_mode(analysis_mode)

    try:
        if not grid_image.filename:
            raise ValueError("Upload one image that contains a grid of visual options.")
        suffix = Path(grid_image.filename).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Upload a normal image file: PNG, JPG, JPEG, or WEBP.")
        report_media_dir.mkdir(parents=True, exist_ok=True)
        upload_path = report_media_dir / f"visual-grid-upload{suffix}"
        upload_path.write_bytes(await grid_image.read())

        visual = build_visual_grid_report(
            upload_path,
            report_media_dir,
            columns=grid_columns,
            rows=grid_rows,
        )
        result = {
            "mode": "visual_grid",
            "source_kind": "visual_grid",
            "report_id": report_id,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "title": Path(grid_image.filename).stem or "Visual grid",
            "visual_grid": {
                "original_url": f"/media/{report_id}/assets/{visual.original_path.name}",
                "heatmap_url": f"/media/{report_id}/assets/{visual.heatmap_path.name}",
                "download_name": visual.heatmap_path.name,
                "summary": visual.summary,
                "cells": _with_visual_asset_urls(report_id, visual.cells),
                "top_picks": _with_visual_asset_urls(report_id, visual.top_picks),
                "recommendations": visual.recommendations,
            },
            "analysis_mode": {"key": selected_analysis_mode, "label": selected_analysis_mode},
        }
        _store_report(report_id, result)
        return _render_page(
            request,
            result=_get_localized_report(result, language),
            error=None,
            language=language,
        )
    except Exception as exc:
        return _render_page(
            request,
            result=None,
            error=_format_error(exc, language),
            language=language,
            status_code=500,
        )


def _render_page(
    request: Request,
    result: dict | None,
    error: str | None,
    language: str,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "TRIBE Review MVP",
            "result": result,
            "error": error,
            "language": language,
            "page_base_url": (result or {}).get("report_page_url", "/"),
            "ui": get_ui_texts(language),
        },
        status_code=status_code,
    )


def _render_pdf_html(report: dict, language: str) -> str:
    template = templates.env.get_template("index.html")
    return template.render(
        {
            "title": "TRIBE Review MVP",
            "result": report,
            "error": None,
            "language": language,
            "page_base_url": report.get("report_page_url", "/"),
            "ui": get_ui_texts(language),
            "pdf_mode": True,
        }
    )


async def _analyze_upload(
    upload: UploadFile,
    report_id: str,
    report_media_dir: Path,
    variant_key: str,
    analysis_mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    suffix = Path(upload.filename or "input.mp4").suffix or ".mp4"
    target_path = report_media_dir / f"{variant_key}{suffix}"
    report_media_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(await upload.read())

    variant_name = Path(upload.filename or target_path.name).stem
    return _analyze_media_path(
        media_path=target_path,
        report_id=report_id,
        variant_key=variant_key,
        variant_name=variant_name,
        analysis_mode=analysis_mode,
    )


def _analyze_media_path(
    media_path: Path,
    report_id: str,
    variant_key: str,
    variant_name: str,
    analysis_mode: str,
    skip_speech: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run = backend.predict_video(media_path)
    result = generate_official_report(
        media_path,
        run,
        variant_name=variant_name,
    )
    result["variant_key"] = variant_key
    result["media_url"] = f"/media/{report_id}/{variant_key}"
    result["_media_path"] = str(media_path)
    result["brain_simulation"] = build_brain_simulation(run.preds, run.timestamps)
    result["report_id"] = report_id
    result["created_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    editorial = _build_editorial_layer(
        media_path,
        run,
        variant_name=variant_name,
        official_result=result,
        analysis_mode=analysis_mode,
        skip_speech=skip_speech,
    )
    if editorial:
        result["editorial"] = editorial
        result["analysis_mode"] = deepcopy(editorial.get("analysis_mode"))
    else:
        result["analysis_mode"] = {
            "key": analysis_mode,
            "label": analysis_mode,
        }
    return result, editorial


def _get_localized_report(report: dict, language: str) -> dict:
    lang = normalize_report_language(language)
    if report.get("mode") == "compare":
        return localize_report(_refresh_comparison_report(report), lang)
    localized = deepcopy(report)
    editorial = localized.get("editorial")
    if isinstance(editorial, dict):
        _sync_editorial_to_official_curve(editorial, localized)
        localized["editorial"] = localize_report(editorial, lang)
    report_id = localized.get("report_id")
    if report_id:
        localized["report_page_url"] = f"/reports/{report_id}"
        localized["report_url"] = f"/reports/{report_id}.json?lang={lang}"
        localized["report_pdf_url"] = f"/reports/{report_id}.pdf?lang={lang}"
    localized["report_language"] = lang
    return localized


def _website_variant_name(url: str) -> str:
    value = (url or "").strip()
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    value = value.split("/", 1)[0].strip() or "Website review"
    return value


async def _build_website_variant(url: str, report_id: str, report_media_dir: Path, index: int) -> dict[str, Any]:
    key = f"site{index}"
    captures = {}
    for device in ("desktop", "mobile"):
        viewport = get_viewport_config(device)
        asset_prefix = f"{key}-{device}"
        screenshot_path = await capture_website_screenshot(
            url,
            report_media_dir,
            name=f"{asset_prefix}-screenshot",
            viewport=device,
        )
        heatmap_path = report_media_dir / f"{asset_prefix}-heatmap.jpg"
        attention = build_website_attention_report(
            screenshot_path,
            heatmap_path,
            viewport_height=int(viewport["height"]),
        )
        captures[device] = {
            "device": device,
            "label": str(viewport["label"]),
            "viewport_width": int(viewport["width"]),
            "viewport_height": int(viewport["height"]),
            "screenshot_url": f"/media/{report_id}/website/{asset_prefix}-screenshot.png",
            "heatmap_url": f"/media/{report_id}/website/{asset_prefix}-heatmap.jpg",
            "download_name": f"{asset_prefix}-heatmap.jpg",
            "summary": attention.summary,
            "sections": attention.sections,
            "recommendations": attention.recommendations,
        }
    return {
        "key": key,
        "label": "Primary URL" if index == 1 else "Comparison URL",
        "title": _website_variant_name(url),
        "url": url,
        "captures": captures,
    }


def _build_website_comparison(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(variants) < 2:
        return None
    rows = []
    for device in ("desktop", "mobile"):
        first = variants[0]["captures"][device]["summary"]
        second = variants[1]["captures"][device]["summary"]
        delta = int(second.get("avg_score", 0)) - int(first.get("avg_score", 0))
        rows.append(
            {
                "device": device,
                "label": variants[0]["captures"][device]["label"],
                "first_score": first.get("avg_score", 0),
                "second_score": second.get("avg_score", 0),
                "delta": delta,
                "winner": variants[1]["title"] if delta > 0 else variants[0]["title"] if delta < 0 else "Even",
            }
        )
    return {"rows": rows}


def _with_visual_asset_urls(report_id: str, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for cell in cells:
        item = deepcopy(cell)
        crop_name = item.get("crop_name")
        if crop_name:
            item["crop_url"] = f"/media/{report_id}/assets/{crop_name}"
        enriched.append(item)
    return enriched


def _refresh_comparison_report(report: dict) -> dict:
    variants = [item for item in report.get("variants", []) if isinstance(item, dict)]
    if len(variants) < 2:
        return deepcopy(report)
    analysis_mode = report.get("analysis_mode") if isinstance(report.get("analysis_mode"), dict) else {}
    refreshed = generate_comparison_report(deepcopy(variants), analysis_mode=analysis_mode.get("key") or DEFAULT_ANALYSIS_MODE)
    for key in ("report_id", "created_at", "official_sources"):
        if key in report:
            refreshed[key] = deepcopy(report[key])
    refreshed["timeline_overlay"] = _build_timeline_overlay(refreshed.get("variants", []))
    return refreshed


def _build_timeline_overlay(variants: list[dict[str, Any]]) -> dict[str, Any]:
    palette = ["#5db0ff", "#75e08c", "#f6b55a", "#df7cff"]
    max_seconds = 0.0
    for variant in variants:
        points = ((variant.get("timeline") or {}).get("points") or []) if isinstance(variant, dict) else []
        for point in points:
            if isinstance(point, dict):
                max_seconds = max(max_seconds, float(point.get("seconds") or 0.0))
    max_seconds = max(max_seconds, 1.0)

    series: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            continue
        points = (variant.get("timeline") or {}).get("points") or []
        svg_points: list[str] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            seconds = float(point.get("seconds") or 0.0)
            score = max(0.0, min(100.0, float(point.get("signal_score") or 0.0)))
            x = 18 + (seconds / max_seconds) * 824
            y = 192 - (score / 100.0) * 174
            svg_points.append(f"{x:.2f},{y:.2f}")
        series.append(
            {
                "variant_key": variant.get("variant_key") or f"v{index + 1}",
                "name": variant.get("title") or variant.get("variant_name") or f"Version {index + 1}",
                "color": palette[index % len(palette)],
                "points": " ".join(svg_points),
                "avg_score": (variant.get("timeline") or {}).get("avg_score"),
                "max_score": (variant.get("timeline") or {}).get("max_score"),
            }
        )
    return {
        "duration_seconds": round(max_seconds, 2),
        "series": series,
    }


def _store_report(report_id: str, report: dict) -> None:
    REPORTS[report_id] = report
    REPORTS.move_to_end(report_id)
    _write_report_file(report_id, report)
    while len(REPORTS) > MAX_REPORTS:
        REPORTS.popitem(last=False)


def _get_stored_report(report_id: str) -> dict | None:
    report = REPORTS.get(report_id)
    if report is not None:
        return report

    report_path = _report_json_path(report_id)
    if not report_path.exists():
        return None
    try:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict):
        media_path = _find_media_file(report_id, "v1")
        if media_path:
            loaded["_media_path"] = media_path
        REPORTS[report_id] = loaded
        REPORTS.move_to_end(report_id)
        return loaded
    return None


def _write_report_file(report_id: str, report: dict) -> None:
    report_path = _report_json_path(report_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_public_report(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _report_json_path(report_id: str) -> Path:
    return MEDIA_DIR / report_id / REPORT_JSON_NAME


def _find_media_file(report_id: str, variant_key: str) -> str | None:
    media_dir = MEDIA_DIR / report_id
    if not media_dir.exists():
        return None
    matches = sorted(media_dir.glob(f"{variant_key}.*"))
    for path in matches:
        if path.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
            return str(path)
    return None


def _public_report(value):
    if isinstance(value, dict):
        return {key: _public_report(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    return value


def _build_editorial_layer(
    video_path: Path,
    run,
    variant_name: str,
    official_result: dict[str, Any] | None = None,
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
    skip_speech: bool = False,
) -> dict | None:
    speech = None
    speech_error = None
    if not skip_speech:
        try:
            speech = speech_backend.transcribe(video_path, analysis_mode=analysis_mode)
        except Exception as exc:
            speech_error = str(exc).strip()

    try:
        review = generate_review(
            video_path,
            run,
            speech=speech,
            speech_error=speech_error,
            analysis_mode=analysis_mode,
            variant_name=variant_name,
        )
    except Exception:
        return None

    if official_result:
        _seed_editorial_curve_points(review, official_result)

    try:
        review = simplify_review_copy(review)
    except Exception:
        pass

    if official_result:
        _sync_editorial_to_official_curve(review, official_result)
    return review


def _normalize_analysis_mode(value: str | None) -> str:
    return DEFAULT_ANALYSIS_MODE

def _seed_editorial_curve_points(review: dict[str, Any], official_result: dict[str, Any]) -> None:
    reference_point, dip_points = _extract_official_curve_points(official_result)
    if not reference_point and not dip_points:
        return
    review["focus_windows"] = _build_curve_focus_windows(
        reference_point,
        dip_points,
        [item for item in review.get("focus_windows", []) if isinstance(item, dict)],
    )
    review["drop_moments"] = _build_curve_drop_moments(
        dip_points,
        [item for item in review.get("drop_moments", []) if isinstance(item, dict)],
    )


def _sync_editorial_to_official_curve(review: dict[str, Any], official_result: dict[str, Any]) -> None:
    reference_point, dip_points = _extract_official_curve_points(official_result)
    if not reference_point and not dip_points:
        return

    review["focus_windows"] = _build_curve_focus_windows(
        reference_point,
        dip_points,
        [item for item in review.get("focus_windows", []) if isinstance(item, dict)],
    )
    review["drop_moments"] = _build_curve_drop_moments(
        dip_points,
        [item for item in review.get("drop_moments", []) if isinstance(item, dict)],
    )
    _rebase_action_items_to_curve(review, reference_point, dip_points)
    _rebuild_editorial_lists(review)
    review["seek_targets"] = _build_editorial_seek_targets(review)


def _extract_official_curve_points(
    official_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    timeline = official_result.get("timeline") if isinstance(official_result.get("timeline"), dict) else {}
    points = [item for item in timeline.get("points", []) if isinstance(item, dict)]
    if not points:
        return None, []

    timestamps = np.asarray([float(item.get("seconds") or 0.0) for item in points], dtype=float)
    scores = np.asarray([float(item.get("signal_score") or 0.0) for item in points], dtype=float)
    if not len(timestamps) or not len(scores):
        return None, []

    video = official_result.get("video") if isinstance(official_result.get("video"), dict) else {}
    duration = float(video.get("duration_seconds") or (timestamps[-1] if len(timestamps) else 0.0))
    start_cutoff = 3.0 if duration > 8.0 else min(1.0, max(0.35, duration * 0.12))
    tail_buffer = 5.0 if duration > 8.0 else max(1.0, duration * 0.30)
    edit_cutoff = duration - tail_buffer if duration > 0 else (timestamps[-1] if len(timestamps) else 0.0)
    valid_edit_indices = [
        index
        for index, second in enumerate(timestamps)
        if start_cutoff < second < edit_cutoff
    ]
    if not valid_edit_indices:
        valid_edit_indices = list(range(len(timestamps)))

    strong_windows = _pick_curve_windows(timestamps, scores, prefer="high", count=2, allowed_indices=valid_edit_indices)
    if not strong_windows:
        strong_windows = _pick_curve_windows(timestamps, scores, prefer="high", count=2)
    weak_windows = _pick_curve_windows(timestamps, scores, prefer="low", count=4, allowed_indices=valid_edit_indices)
    weak_windows = _filter_meaningful_curve_dips(scores, weak_windows, valid_edit_indices)
    reference_point = _curve_point_from_window(timestamps, scores, strong_windows[0]) if strong_windows else None
    dip_points = [_curve_point_from_window(timestamps, scores, item) for item in weak_windows]
    if reference_point:
        dip_points = [item for item in dip_points if item["center_index"] != reference_point["center_index"]]
    return reference_point, dip_points


def _filter_meaningful_curve_dips(
    scores: np.ndarray,
    weak_windows: list[dict[str, Any]],
    valid_indices: list[int],
) -> list[dict[str, Any]]:
    if not weak_windows or not len(scores):
        return []

    valid_scores = scores[valid_indices] if valid_indices else scores
    if not len(valid_scores):
        valid_scores = scores
    working_mean = float(np.mean(valid_scores))
    working_max = float(np.max(valid_scores))

    meaningful: list[dict[str, Any]] = []
    for window in weak_windows:
        center = int(window.get("center_index") or 0)
        score = float(window.get("score") or scores[center])
        local_start = max(0, center - 2)
        local_end = min(len(scores), center + 3)
        local_peak = float(np.max(scores[local_start:local_end])) if local_end > local_start else score

        is_low_absolute = score <= 55.0
        is_local_drop = score <= 65.0 and (local_peak - score) >= 12.0
        is_weak_for_this_cut = score <= (working_mean - 15.0) and (working_max - score) >= 20.0
        if is_low_absolute or is_local_drop or is_weak_for_this_cut:
            meaningful.append(window)

    return meaningful


def _pick_curve_windows(
    timestamps: np.ndarray,
    scores: np.ndarray,
    prefer: str,
    count: int,
    allowed_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not len(scores):
        return []

    allowed = set(allowed_indices) if allowed_indices is not None else set(range(len(scores)))
    centers = sorted(allowed) if allowed else list(range(len(scores)))
    if not centers:
        return []

    window_size = max(3, min(8, len(scores) // 8 or 3))
    scored_windows: list[tuple[float, int, int, int]] = []
    for center in centers:
        half = window_size // 2
        raw_start = max(0, center - half)
        raw_end = min(len(scores) - 1, center + half)
        covered = [index for index in range(raw_start, raw_end + 1) if index in allowed]
        if not covered:
            continue
        start_index = covered[0]
        end_index = covered[-1]
        score = float(np.mean(scores[covered]))
        scored_windows.append((score, center, start_index, end_index))

    scored_windows.sort(key=lambda item: item[0], reverse=(prefer == "high"))
    selected: list[dict[str, Any]] = []
    for score, center, start_index, end_index in scored_windows:
        overlaps = any(abs(center - int(item["center_index"])) <= window_size for item in selected)
        if overlaps:
            continue
        selected.append(
            {
                "score": round(score, 1),
                "center_index": int(center),
                "start_index": int(start_index),
                "end_index": int(end_index),
            }
        )
        if len(selected) >= count:
            break
    return selected


def _curve_point_from_window(
    timestamps: np.ndarray,
    scores: np.ndarray,
    window: dict[str, Any],
) -> dict[str, Any]:
    center_index = int(window["center_index"])
    seconds = round(float(timestamps[center_index]), 2)
    return {
        "seconds": seconds,
        "timestamp": _format_editorial_timestamp(seconds),
        "score": round(float(window.get("score") or scores[center_index]), 1),
        "center_index": center_index,
        "start_seconds": round(float(timestamps[int(window["start_index"])]), 2),
        "end_seconds": round(float(timestamps[int(window["end_index"])]), 2),
    }


def _build_curve_focus_windows(
    reference_point: dict[str, Any] | None,
    dip_points: list[dict[str, Any]],
    existing_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seeds = [point for point in [reference_point, *dip_points[:2]] if point]
    if not seeds:
        return existing_windows

    default_labels = ("Лучший кусок", "Где чинить первым", "Еще одна просадка")
    default_summaries = (
        "Используй этот участок как ориентир.",
        "На графике здесь виден заметный спад.",
        "Здесь на графике есть еще один заметный спад.",
    )
    rewritten: list[dict[str, Any]] = []
    for index, point in enumerate(seeds):
        template = existing_windows[index] if index < len(existing_windows) else {}
        rewritten.append(
            {
                "label": str(template.get("label") or default_labels[index]),
                "timestamp": point["timestamp"],
                "seconds": point["seconds"],
                "summary": str(template.get("summary") or default_summaries[index]),
            }
        )
    return rewritten


def _build_curve_drop_moments(
    dip_points: list[dict[str, Any]],
    existing_drops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for index, point in enumerate(dip_points[:4]):
        template = existing_drops[index] if index < len(existing_drops) else {}
        rewritten.append(
            {
                "seconds": point["seconds"],
                "timestamp": point["timestamp"],
                "reason": str(template.get("reason") or "На графике здесь виден заметный спад."),
            }
        )
    return rewritten


def _rebase_action_items_to_curve(
    review: dict[str, Any],
    reference_point: dict[str, Any] | None,
    dip_points: list[dict[str, Any]],
) -> None:
    actions = [deepcopy(item) for item in review.get("action_items", []) if isinstance(item, dict)]
    if not actions:
        return

    dip_timestamps = [str(item["timestamp"]) for item in dip_points if item.get("timestamp")]
    reference_ts = str(reference_point["timestamp"]) if reference_point and reference_point.get("timestamp") else ""
    updated: list[dict[str, Any]] = []
    dip_index = 0

    for item in actions:
        title = str(item.get("title") or "").strip().lower()
        is_keep = title in {"оставить как есть", "keep as is"}
        if is_keep:
            if reference_ts:
                item["timestamp"] = reference_ts
            updated.append(item)
            continue
        if dip_index >= len(dip_timestamps):
            continue
        item["timestamp"] = dip_timestamps[dip_index]
        dip_index += 1
        updated.append(item)

    review["action_items"] = updated[:4]


def _rebuild_editorial_lists(review: dict[str, Any]) -> None:
    actions = [item for item in review.get("action_items", []) if isinstance(item, dict)]
    if not actions:
        return

    keep_item = next((item for item in actions if _is_keep_action(item)), None)
    edit_items = [item for item in actions if not _is_keep_action(item)]

    existing_strengths = [item for item in review.get("strengths", []) if isinstance(item, str) and item.strip()]
    strengths: list[str] = []
    if keep_item:
        strengths.append(_timed_instruction_line(keep_item))
    extra_strength = next((item for item in existing_strengths if not _looks_timed_line(item)), "")
    if extra_strength:
        strengths.append(extra_strength)
    if strengths:
        review["strengths"] = strengths[:2]

    if edit_items:
        review["weaknesses"] = [_timed_instruction_line(item) for item in edit_items[:2]]

    plan: list[dict[str, str]] = []
    if keep_item:
        plan.append(
            {
                "title": "Оставить",
                "detail": _timed_instruction_line(keep_item),
            }
        )
    if edit_items:
        plan.append(
            {
                "title": "Сделать первым",
                "detail": _timed_instruction_line(edit_items[0]),
            }
        )
    if len(edit_items) > 1:
        plan.append(
            {
                "title": "Сделать потом",
                "detail": _timed_instruction_line(edit_items[1]),
            }
        )
    if plan:
        review["recommendation_plan"] = plan[:3]


def _build_editorial_seek_targets(review: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in review.get("focus_windows", []):
        if isinstance(item, dict):
            targets.append(
                {
                    "label": str(item.get("label") or ""),
                    "timestamp": str(item.get("timestamp") or ""),
                    "seconds": item.get("seconds"),
                    "kind": "focus",
                    "summary": str(item.get("summary") or ""),
                }
            )
    for item in review.get("drop_moments", []):
        if isinstance(item, dict):
            targets.append(
                {
                    "label": "Подозрительный момент",
                    "timestamp": str(item.get("timestamp") or ""),
                    "seconds": item.get("seconds"),
                    "kind": "drop",
                    "summary": str(item.get("reason") or ""),
                }
            )
    speech = review.get("speech")
    if isinstance(speech, dict):
        for segment in speech.get("segments", [])[:6]:
            if not isinstance(segment, dict):
                continue
            start = segment.get("start")
            if not isinstance(start, (int, float)):
                continue
            targets.append(
                {
                    "label": "Speech segment",
                    "timestamp": _format_editorial_timestamp(float(start)),
                    "seconds": round(float(start), 2),
                    "kind": "speech",
                    "summary": str(segment.get("text") or ""),
                }
            )
    return targets


def _is_keep_action(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip().lower()
    return (
        title in {"оставить как есть", "keep as is"}
        or "сохран" in title
        or "keep" in title
    )


def _timed_instruction_line(item: dict[str, Any]) -> str:
    timestamp = str(item.get("timestamp") or "").strip()
    instruction = _compact_editorial_text(str(item.get("instruction") or ""))
    return f"{timestamp}: {instruction}" if timestamp else instruction


def _compact_editorial_text(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip(" -,:.")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def _looks_timed_line(text: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}(?:\s*-\s*\d{2}:\d{2})?\b", str(text).strip()))


def _format_editorial_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_error(exc: Exception, language: str) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    is_llama_gate_error = (
        "meta-llama/llama-3.2-3b" in lowered
        or "trying to access a gated repo" in lowered
        or "access to model meta-llama/llama-3.2-3b is restricted" in lowered
        or "401 client error" in lowered
    )
    if not is_llama_gate_error:
        return message

    if language == "ru":
        return (
            "Официальный TRIBE v2 уперся в gated text encoder из Hugging Face: "
            "meta-llama/Llama-3.2-3B.\n\n"
            "Что нужно сделать:\n"
            "1. Открыть https://huggingface.co/meta-llama/Llama-3.2-3B и запросить доступ.\n"
            "2. Создать read token в Hugging Face.\n"
            "3. Выполнить в PowerShell: huggingface-cli login\n"
            "4. Вставить token и перезапустить приложение.\n\n"
            "Это соответствует официальному workflow TRIBE v2: в опубликованном конфиге "
            "text encoder использует gated LLaMA 3.2-3B."
        )

    return (
        "Official TRIBE v2 hit a gated Hugging Face text encoder: "
        "meta-llama/Llama-3.2-3B.\n\n"
        "What you need to do:\n"
        "1. Open https://huggingface.co/meta-llama/Llama-3.2-3B and request access.\n"
        "2. Create a read token in Hugging Face.\n"
        "3. Run in PowerShell: huggingface-cli login\n"
        "4. Paste the token and restart the app.\n\n"
        "This matches the official TRIBE v2 workflow: the published config uses gated "
        "LLaMA 3.2-3B as the text encoder."
    )
