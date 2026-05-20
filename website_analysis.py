from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class WebsiteAttentionResult:
    heatmap_path: Path
    sections: list[dict[str, Any]]
    summary: dict[str, Any]
    recommendations: list[str]


def build_website_attention_report(
    screenshot_path: Path,
    heatmap_path: Path,
    viewport_height: int = 1200,
) -> WebsiteAttentionResult:
    image = Image.open(screenshot_path).convert("RGB")
    heat = _attention_heat(image)
    overlay = _overlay_heatmap(image, heat)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(heatmap_path, quality=92)

    sections = _section_summaries(heat, image.size, viewport_height=viewport_height)
    summary = _summary(sections, image.size, viewport_height=viewport_height)
    return WebsiteAttentionResult(
        heatmap_path=heatmap_path,
        sections=sections,
        summary=summary,
        recommendations=_recommendations(sections),
    )


def _attention_heat(image: Image.Image) -> np.ndarray:
    # Downscale before analysis to keep full-page captures cheap.
    width, height = image.size
    scale = min(1.0, 1000.0 / max(width, 1))
    working = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    rgb = np.asarray(working).astype(np.float32) / 255.0
    gray = np.asarray(working.convert("L")).astype(np.float32) / 255.0

    blur_small = np.asarray(working.convert("L").filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255.0
    blur_large = np.asarray(working.convert("L").filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255.0
    contrast = np.abs(blur_small - blur_large)

    gy, gx = np.gradient(gray)
    edges = np.sqrt(gx * gx + gy * gy)

    colorfulness = rgb.max(axis=2) - rgb.min(axis=2)

    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    x_norm = xx / max(gray.shape[1] - 1, 1)
    y_norm = yy / max(gray.shape[0] - 1, 1)
    center_bias = np.exp(-((x_norm - 0.5) ** 2) / 0.12)
    fold_bias = 0.72 + 0.28 * np.exp(-((y_norm % 1.0) ** 2) / 0.45)

    heat = contrast * 0.42 + edges * 0.34 + colorfulness * 0.18 + center_bias * fold_bias * 0.06
    heat = _normalize(heat)
    heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10))).astype(np.float32) / 255.0
    return _normalize(heat)


def _overlay_heatmap(image: Image.Image, heat: np.ndarray) -> Image.Image:
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BICUBIC)
    heat_full = np.asarray(heat_img).astype(np.float32) / 255.0
    base = np.asarray(image).astype(np.float32) / 255.0
    colors = _heat_colors(heat_full)
    alpha = np.clip((heat_full - 0.24) / 0.58, 0, 1) ** 1.45
    alpha = alpha[..., None] * 0.82
    mixed = base * (1 - alpha) + colors * alpha
    return Image.fromarray((np.clip(mixed, 0, 1) * 255).astype(np.uint8))


def _heat_colors(heat: np.ndarray) -> np.ndarray:
    cold = np.array([0.0, 0.36, 1.0], dtype=np.float32)
    mid = np.array([1.0, 0.82, 0.0], dtype=np.float32)
    hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    peak = np.array([0.72, 0.0, 1.0], dtype=np.float32)
    t = heat[..., None]
    first_t = np.clip(t / 0.48, 0, 1)
    second_t = np.clip((t - 0.48) / 0.34, 0, 1)
    peak_t = np.clip((t - 0.82) / 0.18, 0, 1)
    first = cold * (1 - first_t) + mid * first_t
    second = mid * (1 - second_t) + hot * second_t
    third = hot * (1 - peak_t) + peak * peak_t
    return np.where(t < 0.48, first, np.where(t < 0.82, second, third))


def _section_summaries(heat: np.ndarray, image_size: tuple[int, int], viewport_height: int) -> list[dict[str, Any]]:
    width, height = image_size
    scale_y = heat.shape[0] / max(height, 1)
    sections: list[dict[str, Any]] = []
    count = max(1, int(np.ceil(height / max(viewport_height, 1))))
    for index in range(count):
        y0 = index * viewport_height
        y1 = min(height, (index + 1) * viewport_height)
        h0 = int(y0 * scale_y)
        h1 = max(h0 + 1, int(y1 * scale_y))
        section = heat[h0:h1, :]
        score = float(np.mean(section))
        intensity = int(round(score * 100))
        strongest = np.unravel_index(int(np.argmax(section)), section.shape)
        focus_x = int((strongest[1] / max(section.shape[1] - 1, 1)) * width)
        focus_y = int(y0 + (strongest[0] / max(section.shape[0] - 1, 1)) * max(y1 - y0, 1))
        sections.append(
            {
                "label": "Above the fold" if index == 0 else f"Scroll section {index + 1}",
                "range": f"{y0}px - {y1}px",
                "y0": y0,
                "y1": y1,
                "top_percent": round((y0 / max(height, 1)) * 100, 3),
                "height_percent": round(((y1 - y0) / max(height, 1)) * 100, 3),
                "score": intensity,
                "summary": _section_copy(index, intensity),
                "focus": {"x": focus_x, "y": focus_y},
            }
        )
    return sections


def _section_copy(index: int, score: int) -> str:
    if index == 0 and score >= 48:
        return "The first screen has a clear attention cluster. Check whether that cluster supports the main offer and CTA."
    if index == 0:
        return "The first screen looks visually diffuse. The main offer or CTA may need a stronger visual anchor."
    if score >= 48:
        return "This scroll section has a strong visual pull. It can carry proof, product value, or a secondary CTA."
    return "This scroll section is quieter. Use it for supporting detail or simplify it if it hides an important action."


def _summary(sections: list[dict[str, Any]], image_size: tuple[int, int], viewport_height: int) -> dict[str, Any]:
    if not sections:
        return {}
    strongest = max(sections, key=lambda item: item["score"])
    weakest = min(sections, key=lambda item: item["score"])
    avg_score = int(round(float(np.mean([item["score"] for item in sections]))))
    return {
        "page_size": f"{image_size[0]}x{image_size[1]}",
        "width": image_size[0],
        "height": image_size[1],
        "viewport_height": viewport_height,
        "sections_count": len(sections),
        "strongest_section": strongest["label"],
        "weakest_section": weakest["label"],
        "avg_score": avg_score,
        "strongest_score": strongest["score"],
        "weakest_score": weakest["score"],
        "headline": "Predicted attention map",
        "note": "This is not eye tracking. It is a visual attention estimate based on contrast, edges, color, layout position, and page structure.",
    }


def _recommendations(sections: list[dict[str, Any]]) -> list[str]:
    if not sections:
        return ["Capture the page again or try another URL."]

    first = sections[0]
    strongest = max(sections, key=lambda item: item["score"])
    weakest = min(sections, key=lambda item: item["score"])
    recs: list[str] = []

    if first["score"] < 45:
        recs.append("Strengthen the first screen: make the main promise, hero visual, or CTA more visually dominant above the fold.")
    else:
        recs.append("Keep the first screen as the main anchor, but check that the hottest zone supports the offer and not a decorative element.")

    if strongest["label"] != first["label"]:
        recs.append(f"The strongest pull is lower on the page ({strongest['label']}). Consider moving that proof, visual, or offer closer to the first screen.")

    if weakest["score"] < 28:
        recs.append(f"{weakest['label']} is visually quiet. If it contains an important CTA or sales argument, simplify the layout or increase contrast.")

    if len(sections) > 2:
        recs.append("Use the fold markers to check whether every scroll section has one clear job: offer, proof, CTA, or detail.")

    return recs[:4]


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 98))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
