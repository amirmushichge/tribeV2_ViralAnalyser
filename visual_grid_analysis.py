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

    cells = _build_cells(image, overlay, heat, output_dir, columns, rows)
    ranked = sorted(cells, key=lambda item: item["score"], reverse=True)
    for rank, cell in enumerate(ranked, start=1):
        cell["rank"] = rank
    for cell in ranked:
        cell["reasons"] = _reasons(cell, len(ranked), rows, columns)
        cell["risk"] = _risk(cell)
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
    overlay: Image.Image,
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
            heatmap_crop_path = output_dir / f"visual-cell-{index:02d}-heatmap.jpg"
            image.crop(crop_box).save(crop_path, quality=92)
            overlay.crop(crop_box).save(heatmap_crop_path, quality=92)
            cells.append(
                {
                    "index": index,
                    "label": f"Visual {index}",
                    "row": row + 1,
                    "column": col + 1,
                    "score": score,
                    "crop_name": crop_path.name,
                    "heatmap_crop_name": heatmap_crop_path.name,
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
                    "reasons": [],
                    "risk": "",
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


def _reasons(cell: dict[str, Any], total: int, rows: int, columns: int) -> list[str]:
    metrics = cell["metrics"]
    score = int(cell["score"])
    rank = int(cell["rank"])
    position = _position_name(int(cell["row"]), int(cell["column"]), rows, columns)
    reasons: list[str] = [_rank_reason(rank, total, score, position)]

    if metrics["attention_peak"] >= 0.72:
        reasons.append(_pick(cell, "The hottest area is easy to find, so the first read has a real anchor.", "A clear heat spike gives the viewer somewhere obvious to land.", "The main pull is not vague; one zone wins attention quickly."))
    elif metrics["attention_peak"] >= 0.55:
        reasons.append(_pick(cell, "It gets noticed, but the hotspot is not dominant enough to feel like a knockout.", "The read is decent; the main subject still has to work for attention.", "There is visible pull here, just not a runaway one."))
    else:
        reasons.append(_pick(cell, "The eye does not get a strong first target here.", "This option may be readable, but it does not interrupt the scan strongly.", "The heat is too soft to make this a confident first pick."))

    if metrics["focus"] >= 0.24:
        reasons.append(_pick(cell, "Attention is concentrated, which is useful for thumbnails and fast feeds.", "The visual has a focused landing zone instead of scattering the viewer.", "The composition gives one main place to look first."))
    elif metrics["edge_density"] >= 0.24:
        reasons.append(_pick(cell, "There is a lot going on, so the core idea may fight with details.", "Small details create texture, but they also dilute the first read.", "The design may need a cleaner hero element before launch."))
    elif metrics["contrast"] >= 0.22:
        reasons.append(_pick(cell, "Contrast helps it stay readable when the grid is scanned fast.", "The subject separates well enough from the background.", "Shape and value separation are doing useful work here."))
    elif metrics["colorfulness"] >= 0.28:
        reasons.append(_pick(cell, "Color gives it personality, even if the attention peak is not the strongest.", "The palette helps it stand apart from quieter options.", "Color is carrying part of the first impression."))
    else:
        reasons.append(_pick(cell, "It needs a sharper subject, stronger value contrast, or a cleaner focal point.", "The idea may be fine, but the visual signal is too polite.", "This one would benefit from a bolder foreground/background split."))

    if metrics["center_bias"] >= 0.68:
        reasons.append(f"{position.capitalize()} placement supports the natural scan path.")
    elif score < 50:
        reasons.append(f"{position.capitalize()} placement does not rescue the weaker attention pull.")

    return _unique(reasons)[:3]


def _risk(cell: dict[str, Any]) -> str:
    metrics = cell["metrics"]
    score = int(cell["score"])
    rank = int(cell["rank"])
    if metrics["edge_density"] >= 0.3:
        return "Risk: visual noise may compete with the main subject."
    if metrics["contrast"] < 0.13 and metrics["colorfulness"] < 0.15:
        return "Risk: the image may feel flat next to stronger options."
    if metrics["attention_peak"] < 0.42:
        return "Risk: easy to skip because no strong hotspot was found."
    if rank == 1:
        return "Best use: lead creative or first A/B test candidate."
    if score >= 55:
        return "Best use: backup test if the message fits this audience better."
    return "Best use: support option after simplifying the focal point."


def _rank_reason(rank: int, total: int, score: int, position: str) -> str:
    if rank == 1:
        return f"Winner read: {position} pulls attention first in this set."
    if rank == 2 and total > 2:
        return f"Second choice: close enough to test if the message is stronger."
    if rank <= max(3, total // 2):
        if score >= 55:
            return f"Middle contender: useful, but it needs a clearer reason to beat the winner."
        return f"Middle of the pack: visible, but not sharp enough as the lead option."
    if score >= 55:
        options = [
            f"Backup angle: {position} has pull, just not enough to lead the set.",
            f"Safe alternate: the idea reads, but it loses the first-glance fight.",
            f"Usable variant: keep it for testing a different message, not as the main pick.",
        ]
    elif score >= 45:
        options = [
            f"Support visual: {position} is readable, but the hook needs more force.",
            f"Scroll-past risk: it may blend in before the viewer understands the idea.",
            f"Needs a sharper hook: the composition does not create a fast enough stop.",
        ]
    else:
        options = [
            f"Low-pull option: {position} needs a much clearer focal point.",
            f"Easy to miss: the first read is weaker than the rest of the grid.",
            f"Rework candidate: simplify the image before using it as a lead visual.",
        ]
    return options[(rank - 1) % len(options)]


def _position_name(row: int, column: int, rows: int, columns: int) -> str:
    vertical = "top" if row == 1 else "bottom" if row == rows else "middle"
    horizontal = "left" if column == 1 else "right" if column == columns else "center"
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return f"{horizontal} side"
    if horizontal == "center":
        return f"{vertical} center"
    return f"{vertical} {horizontal}"


def _pick(cell: dict[str, Any], *options: str) -> str:
    if not options:
        return ""
    index = int(cell["index"]) - 1
    rank = int(cell.get("rank", 1)) - 1
    return options[(index + rank) % len(options)]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


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
