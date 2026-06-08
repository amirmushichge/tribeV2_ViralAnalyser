from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from website_analysis import _attention_heat, _overlay_heatmap


@dataclass(frozen=True)
class VisualGridResult:
    original_path: Path
    heatmap_path: Path
    cells: list[dict[str, Any]]
    top_picks: list[dict[str, Any]]
    summary: dict[str, Any]
    recommendations: list[str]


COMMON_GRID_SHAPES = [
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
    (4, 3),
    (3, 4),
    (4, 2),
    (2, 4),
    (4, 4),
]


def build_visual_grid_report(
    image_path: Path,
    output_dir: Path,
    columns: int | None = None,
    rows: int | None = None,
) -> VisualGridResult:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    if width < 160 or height < 160:
        raise ValueError("Upload a larger image grid. Very small images do not give a useful read.")

    columns, rows = _resolve_grid_shape(width, height, columns, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "visual-grid-original.png"
    heatmap_path = output_dir / "visual-grid-heatmap.jpg"
    image.save(original_path)

    heat = _attention_heat(image)
    overlay = _overlay_heatmap(image, heat)
    overlay.save(heatmap_path, quality=92)

    cells = _build_cells(image, heat, output_dir, columns, rows)
    ranked = sorted(cells, key=lambda item: item["score"], reverse=True)
    for rank, cell in enumerate(ranked, start=1):
        cell["rank"] = rank
    cells = sorted(ranked, key=lambda item: item["index"])
    top_picks = [cell for cell in ranked[: min(3, len(ranked))]]
    summary = _summary(image.size, columns, rows, ranked)

    return VisualGridResult(
        original_path=original_path,
        heatmap_path=heatmap_path,
        cells=cells,
        top_picks=top_picks,
        summary=summary,
        recommendations=_recommendations(ranked),
    )


def _resolve_grid_shape(
    width: int,
    height: int,
    columns: int | None,
    rows: int | None,
) -> tuple[int, int]:
    clean_columns = _clean_dimension(columns)
    clean_rows = _clean_dimension(rows)
    if clean_columns and clean_rows:
        return clean_columns, clean_rows

    aspect = width / max(height, 1)
    candidates = COMMON_GRID_SHAPES
    if clean_columns:
        candidates = [item for item in candidates if item[0] == clean_columns] or [(clean_columns, row) for row in range(1, 6)]
    if clean_rows:
        candidates = [item for item in candidates if item[1] == clean_rows] or [(col, clean_rows) for col in range(1, 6)]

    def penalty(shape: tuple[int, int]) -> float:
        cols, row_count = shape
        shape_aspect = cols / max(row_count, 1)
        count = cols * row_count
        count_penalty = 0.0 if count in {4, 6, 9, 12} else 0.18
        return abs(np.log(aspect / max(shape_aspect, 0.001))) + count_penalty

    return min(candidates, key=penalty)


def _clean_dimension(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None
    return min(number, 6)


def _build_cells(
    image: Image.Image,
    heat: np.ndarray,
    output_dir: Path,
    columns: int,
    rows: int,
) -> list[dict[str, Any]]:
    width, height = image.size
    heat_height, heat_width = heat.shape
    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    rgb = np.asarray(image).astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)
    cells: list[dict[str, Any]] = []

    for row in range(rows):
        for col in range(columns):
            index = row * columns + col + 1
            x0 = round(col * width / columns)
            x1 = round((col + 1) * width / columns)
            y0 = round(row * height / rows)
            y1 = round((row + 1) * height / rows)
            crop_box = (x0, y0, x1, y1)

            heat_crop = heat[
                max(0, int(y0 * heat_height / height)) : max(1, int(y1 * heat_height / height)),
                max(0, int(x0 * heat_width / width)) : max(1, int(x1 * heat_width / width)),
            ]
            gray_crop = gray[y0:y1, x0:x1]
            rgb_crop = rgb[y0:y1, x0:x1]
            edge_crop = edge[y0:y1, x0:x1]

            metrics = _cell_metrics(heat_crop, gray_crop, rgb_crop, edge_crop, col, row, columns, rows)
            score = _cell_score(metrics)
            crop_path = output_dir / f"visual-cell-{index:02d}.jpg"
            image.crop(crop_box).save(crop_path, quality=92)
            cells.append(
                {
                    "index": index,
                    "label": f"Visual {index}",
                    "row": row + 1,
                    "column": col + 1,
                    "score": score,
                    "crop_name": crop_path.name,
                    "box": {
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "left_percent": round(x0 / max(width, 1) * 100, 4),
                        "top_percent": round(y0 / max(height, 1) * 100, 4),
                        "width_percent": round((x1 - x0) / max(width, 1) * 100, 4),
                        "height_percent": round((y1 - y0) / max(height, 1) * 100, 4),
                    },
                    "metrics": metrics,
                    "reasons": _reasons(metrics),
                    "risk": _risk(metrics),
                }
            )
    return cells


def _cell_metrics(
    heat_crop: np.ndarray,
    gray_crop: np.ndarray,
    rgb_crop: np.ndarray,
    edge_crop: np.ndarray,
    col: int,
    row: int,
    columns: int,
    rows: int,
) -> dict[str, float]:
    heat_mean = float(np.mean(heat_crop)) if heat_crop.size else 0.0
    heat_peak = float(np.percentile(heat_crop, 92)) if heat_crop.size else 0.0
    contrast = float(np.std(gray_crop)) if gray_crop.size else 0.0
    colorfulness = float(np.mean(rgb_crop.max(axis=2) - rgb_crop.min(axis=2))) if rgb_crop.size else 0.0
    edge_density = float(np.mean(edge_crop)) if edge_crop.size else 0.0
    center_x = (col + 0.5) / max(columns, 1)
    center_y = (row + 0.5) / max(rows, 1)
    center_bias = float(np.exp(-(((center_x - 0.5) ** 2) / 0.18 + ((center_y - 0.42) ** 2) / 0.28)))
    focus = float(heat_peak - heat_mean)
    return {
        "attention_mean": round(heat_mean, 4),
        "attention_peak": round(heat_peak, 4),
        "contrast": round(contrast, 4),
        "colorfulness": round(colorfulness, 4),
        "edge_density": round(edge_density, 4),
        "center_bias": round(center_bias, 4),
        "focus": round(focus, 4),
    }


def _cell_score(metrics: dict[str, float]) -> int:
    raw = (
        metrics["attention_mean"] * 0.46
        + metrics["attention_peak"] * 0.24
        + min(metrics["contrast"] * 2.2, 1.0) * 0.12
        + min(metrics["colorfulness"] * 2.4, 1.0) * 0.1
        + metrics["center_bias"] * 0.08
        - max(metrics["edge_density"] - 0.18, 0.0) * 0.16
    )
    return int(round(max(0.0, min(1.0, raw)) * 100))


def _reasons(metrics: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if metrics["attention_peak"] >= 0.72:
        reasons.append("Strong hotspot: the eye has a clear place to land.")
    if metrics["contrast"] >= 0.22:
        reasons.append("Good contrast makes the visual easier to catch fast.")
    if metrics["colorfulness"] >= 0.28:
        reasons.append("Color separates it from nearby options.")
    if metrics["center_bias"] >= 0.68:
        reasons.append("Position is close to the natural scan path of the grid.")
    if metrics["focus"] >= 0.24:
        reasons.append("Attention is concentrated instead of being spread everywhere.")
    if not reasons:
        reasons.append("Readable, but it does not create a strong attention advantage over the grid.")
    return reasons[:3]


def _risk(metrics: dict[str, float]) -> str:
    if metrics["edge_density"] >= 0.3:
        return "May be noisy: too many small details can compete with the main subject."
    if metrics["contrast"] < 0.13 and metrics["colorfulness"] < 0.15:
        return "May feel flat: contrast and color separation are both low."
    if metrics["attention_peak"] < 0.42:
        return "May be easy to skip: no strong hotspot was found."
    return "No major visual risk detected."


def _summary(image_size: tuple[int, int], columns: int, rows: int, ranked: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [item["score"] for item in ranked]
    best = ranked[0] if ranked else {}
    return {
        "image_size": f"{image_size[0]}x{image_size[1]}",
        "columns": columns,
        "rows": rows,
        "cells_count": columns * rows,
        "avg_score": int(round(float(np.mean(scores)))) if scores else 0,
        "top_label": best.get("label", "Visual 1"),
        "top_score": best.get("score", 0),
        "headline": "Visual grid attention review",
        "note": "This is a visual attention pre-check. It estimates which option pulls the eye first; it is not a guarantee of post performance.",
    }


def _recommendations(ranked: list[dict[str, Any]]) -> list[str]:
    if not ranked:
        return ["Upload a grid image with several visual options."]
    best = ranked[0]
    recs = [
        f"Start with {best['label']}: it has the strongest predicted visual pull in this grid.",
    ]
    if len(ranked) > 1:
        second = ranked[1]
        gap = int(best["score"]) - int(second["score"])
        if gap <= 5:
            recs.append(f"{second['label']} is close. Test both if the message or audience is different.")
        else:
            recs.append(f"{best['label']} leads by {gap} points over {second['label']}.")
    noisy = [item for item in ranked if "noisy" in item.get("risk", "").lower()]
    if noisy:
        recs.append(f"Simplify {noisy[-1]['label']} if it contains the main idea; small details may be competing for attention.")
    low = [item for item in ranked if int(item["score"]) < 36]
    if low:
        recs.append("The lowest-scoring options need a clearer subject, stronger contrast, or less visual clutter.")
    return recs[:4]
