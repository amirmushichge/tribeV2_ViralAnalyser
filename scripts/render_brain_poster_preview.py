from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_visualization import (  # noqa: E402
    BRAIN_MESH_LEVEL,
    POSTER_BACKGROUND_RGB,
    POSTER_DPI,
    POSTER_SIZE_INCHES,
    REGION_DEFINITIONS,
    SIGNAL_QUANTIZATION,
    _load_mesh_bundle,
    _polish_surface_poster,
    render_surface_poster,
)


DEFAULT_DURATION_SECONDS = 8.0
DEFAULT_FRAME_COUNT = 24
DEFAULT_PEAK_SECONDS = 6.6
DEFAULT_OUTPUT = Path("/tmp/tribe-brain-poster-preview.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a deterministic TRIBE brain-poster preview without running video inference, "
            "Cloud Tasks, or the full worker pipeline."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--polish-input",
        type=Path,
        help="Apply the poster polish pass to an existing PNG artifact instead of rendering a preview mesh.",
    )
    parser.add_argument(
        "--mesh-source",
        choices=("procedural", "tribev2"),
        default="procedural",
        help="Use the dependency-free procedural mesh by default; use tribev2 only in a full worker environment.",
    )
    parser.add_argument("--assert-quality", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.polish_input:
        args.output.write_bytes(_polish_surface_poster(args.polish_input.read_bytes()))
    else:
        simulation = build_preview_simulation(args.mesh_source)
        args.output.write_bytes(render_surface_poster(simulation, DEFAULT_PEAK_SECONDS))

    report: dict[str, Any] = {"candidate": score_poster(args.output)}
    if args.compare:
        report["compare"] = score_poster(args.compare)
        report["delta"] = {
            key: round(report["candidate"][key] - report["compare"][key], 4)
            for key in (
                "content_width_ratio",
                "content_height_ratio",
                "content_area_ratio",
                "heat_pixel_ratio",
                "mean_luminance",
            )
        }

    if args.assert_quality:
        assert_quality(report["candidate"])

    print(json.dumps(report, indent=2, sort_keys=True))


def build_preview_simulation(mesh_source: str) -> dict[str, Any]:
    mesh_bundle = _preview_mesh_bundle(mesh_source)
    coords = np.asarray(mesh_bundle["surfaces"]["inflated"], dtype=float)
    timestamps = np.linspace(0.0, DEFAULT_DURATION_SECONDS, num=DEFAULT_FRAME_COUNT)
    centers = _preview_centers(coords)

    frames = []
    for index, seconds in enumerate(timestamps):
        signal = _preview_signal(coords, centers, seconds)
        frames.append(
            {
                "index": index,
                "seconds": round(float(seconds), 2),
                "timestamp": _format_seconds(seconds),
                "signal": np.rint(signal * SIGNAL_QUANTIZATION).astype(np.uint8).tolist(),
                "region_scores": [0.0 for _ in REGION_DEFINITIONS],
                "region_peak_scores": [0.0 for _ in REGION_DEFINITIONS],
            }
        )

    return {
        "available": True,
        "frame_count": len(frames),
        "mesh_level": f"{BRAIN_MESH_LEVEL} deterministic preview",
        "mesh": {
            "faces": mesh_bundle["faces"].reshape(-1).astype(int).tolist(),
            "bg_map": np.rint(mesh_bundle["bg_map"] * SIGNAL_QUANTIZATION).astype(np.uint8).tolist(),
            "region_ids": mesh_bundle["region_ids"].astype(np.uint8).tolist(),
            "regions": [
                {
                    "id": int(item["id"]),
                    "key": item["key"],
                    "color": item["color"],
                    "label_en": item["label_en"],
                    "description_en": item["description_en"],
                }
                for item in REGION_DEFINITIONS
            ],
            "surfaces": {
                key: np.round(value.astype(np.float32).reshape(-1), 4).tolist()
                for key, value in mesh_bundle["surfaces"].items()
            },
            "default_surface": "inflated",
        },
        "frames": frames,
    }


def _preview_mesh_bundle(mesh_source: str) -> dict[str, Any]:
    if mesh_source == "tribev2":
        return _load_mesh_bundle(BRAIN_MESH_LEVEL)
    return _procedural_mesh_bundle()


def _procedural_mesh_bundle() -> dict[str, Any]:
    coords: list[list[float]] = []
    bg_values: list[float] = []
    faces: list[list[int]] = []
    rows = 38
    cols = 60

    for sign in (-1.0, 1.0):
        offset = len(coords)
        for row in range(rows):
            theta = np.pi * row / (rows - 1)
            for col in range(cols):
                phi = 2.0 * np.pi * col / cols
                fold = 1.0 + 0.04 * np.sin(9.0 * phi + sign * 0.8) * np.sin(6.0 * theta)
                x = sign * 0.38 + sign * 0.34 * np.sin(theta) * np.cos(phi) * fold
                y = 0.62 * np.sin(theta) * np.sin(phi) * fold
                z = 0.42 * np.cos(theta) * (1.0 + 0.025 * np.sin(7.0 * phi))
                coords.append([x, y, z])
                bg_values.append(0.34 + 0.2 * np.sin(12.0 * phi) * np.sin(5.0 * theta))

        for row in range(rows - 1):
            for col in range(cols):
                a = offset + row * cols + col
                b = offset + row * cols + (col + 1) % cols
                c = offset + (row + 1) * cols + col
                d = offset + (row + 1) * cols + (col + 1) % cols
                faces.append([a, c, b])
                faces.append([b, c, d])

    coords_np = np.asarray(coords, dtype=float)
    bg_np = np.asarray(bg_values, dtype=float)
    bg_np = (bg_np - bg_np.min()) / max(float(bg_np.max() - bg_np.min()), 1e-6)
    return {
        "faces": np.asarray(faces, dtype=int),
        "bg_map": 0.24 + bg_np * 0.14,
        "region_ids": np.zeros(len(coords_np), dtype=np.uint8),
        "region_masks": [np.zeros(len(coords_np), dtype=bool) for _ in REGION_DEFINITIONS],
        "region_meta": [],
        "surfaces": {
            "normal": coords_np,
            "inflated": coords_np,
        },
    }


def _preview_centers(coords: np.ndarray) -> np.ndarray:
    quantiles = (
        (0.2, 0.31, 0.42),
        (0.37, 0.43, 0.6),
        (0.5, 0.24, 0.38),
        (0.63, 0.43, 0.6),
        (0.8, 0.31, 0.42),
    )
    return np.asarray(
        [
            [
                float(np.quantile(coords[:, 0], qx)),
                float(np.quantile(coords[:, 1], qy)),
                float(np.quantile(coords[:, 2], qz)),
            ]
            for qx, qy, qz in quantiles
        ],
        dtype=float,
    )


def _preview_signal(coords: np.ndarray, centers: np.ndarray, seconds: float) -> np.ndarray:
    progress = seconds / DEFAULT_DURATION_SECONDS
    amplitudes = np.asarray(
        [
            0.5 + 0.42 * np.exp(-((progress - 0.22) / 0.22) ** 2),
            0.42 + 0.58 * np.exp(-((progress - 0.58) / 0.2) ** 2),
            0.34 + 0.56 * np.exp(-((progress - 0.72) / 0.18) ** 2),
            0.42 + 0.6 * np.exp(-((progress - 0.77) / 0.15) ** 2),
            0.48 + 0.44 * np.exp(-((progress - 0.88) / 0.16) ** 2),
        ],
        dtype=float,
    )
    widths = np.asarray([0.19, 0.16, 0.2, 0.16, 0.19], dtype=float)

    signal = np.zeros(len(coords), dtype=float)
    for center, amplitude, width in zip(centers, amplitudes, widths, strict=True):
        distance_squared = np.sum((coords - center) ** 2, axis=1)
        signal += amplitude * np.exp(-distance_squared / (2.0 * width * width))

    signal += 0.04 * (1.0 + np.sin(progress * np.pi * 2.0))
    signal = signal / max(float(signal.max()), 1e-6)
    return np.power(signal, 0.82)


def score_poster(path: Path) -> dict[str, Any]:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.uint8)
    rgb = pixels[..., :3].astype(float)
    alpha = pixels[..., 3]
    background = np.asarray(POSTER_BACKGROUND_RGB, dtype=float)
    distance = np.linalg.norm(rgb - background, axis=2)
    luminance = (rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=float)) / 255.0
    content = (alpha > 0) & ((distance > 10.0) | (luminance > 0.12))
    heat = (
        (alpha > 0)
        & (rgb[..., 1] > 90)
        & ((rgb[..., 1] - rgb[..., 0] > 18) | (rgb[..., 0] > 145))
    )

    if bool(content.any()):
        ys, xs = np.where(content)
        width_ratio = (int(xs.max()) - int(xs.min()) + 1) / image.width
        height_ratio = (int(ys.max()) - int(ys.min()) + 1) / image.height
        area_ratio = float(content.mean())
    else:
        width_ratio = 0.0
        height_ratio = 0.0
        area_ratio = 0.0

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "width": image.width,
        "height": image.height,
        "content_width_ratio": round(float(width_ratio), 4),
        "content_height_ratio": round(float(height_ratio), 4),
        "content_area_ratio": round(area_ratio, 4),
        "heat_pixel_ratio": round(float(heat.mean()), 4),
        "mean_luminance": round(float(luminance.mean()), 4),
    }


def assert_quality(metrics: dict[str, Any]) -> None:
    expected_width = int(POSTER_SIZE_INCHES[0] * POSTER_DPI)
    expected_height = int(POSTER_SIZE_INCHES[1] * POSTER_DPI)
    failures = []
    if metrics["width"] != expected_width or metrics["height"] != expected_height:
        actual_size = f"{metrics['width']}x{metrics['height']}"
        failures.append(f"expected {expected_width}x{expected_height}, got {actual_size}")
    if metrics["content_width_ratio"] < 0.58:
        failures.append("cortex content is too narrow")
    if metrics["content_height_ratio"] < 0.26:
        failures.append("cortex content is too short")
    if metrics["heat_pixel_ratio"] < 0.012:
        failures.append("activation heat is too sparse")
    if metrics["mean_luminance"] < 0.045:
        failures.append("poster is too dark")
    if failures:
        raise SystemExit("Poster quality check failed: " + "; ".join(failures))


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    main()
