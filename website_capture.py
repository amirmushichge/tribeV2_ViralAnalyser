from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from runtime_setup import ensure_local_ffmpeg_on_path


VIEWPORTS = {
    "desktop": {"width": 1440, "height": 1200, "label": "Desktop"},
    "mobile": {"width": 390, "height": 844, "label": "Mobile"},
}
DESKTOP_VIEWPORT = VIEWPORTS["desktop"]
VIDEO_SIZE = "1280:720"


async def capture_website_video(url: str, output_dir: Path, name: str = "website") -> tuple[Path, Path]:
    normalized_url = _normalize_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / f"{name}.png"
    video_path = output_dir / f"{name}.mp4"

    await _capture_screenshot(normalized_url, screenshot_path)
    _screenshot_to_video(screenshot_path, video_path)
    return video_path, screenshot_path


async def capture_website_screenshot(url: str, output_dir: Path, name: str = "website", viewport: str = "desktop") -> Path:
    normalized_url = _normalize_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / f"{name}.png"
    await _capture_screenshot(normalized_url, screenshot_path, viewport=viewport)
    return screenshot_path


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError("Paste a website URL to run a website attention review.")
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Use a normal http or https website URL.")
    return cleaned


def get_viewport_config(viewport: str) -> dict[str, int | str]:
    return VIEWPORTS.get(viewport, VIEWPORTS["desktop"])


async def _capture_screenshot(url: str, screenshot_path: Path, viewport: str = "desktop") -> None:
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import async_playwright
    except Exception as exc:
        raise RuntimeError("Website URL mode needs the Playwright Python package. Run Start_TRIBE_Review.cmd again to install dependencies.") from exc

    last_error: Exception | None = None
    viewport_config = get_viewport_config(viewport)
    viewport_size = {"width": int(viewport_config["width"]), "height": int(viewport_config["height"])}
    is_mobile = viewport == "mobile"
    async with async_playwright() as p:
        for browser_name, launch_kwargs in (
            ("Chrome", {"channel": "chrome"}),
            ("Microsoft Edge", {"channel": "msedge"}),
            ("Playwright Chromium", {}),
        ):
            browser = None
            try:
                browser = await p.chromium.launch(headless=True, **launch_kwargs)
                page = await browser.new_page(
                    viewport=viewport_size,
                    device_scale_factor=2 if is_mobile else 1,
                    is_mobile=is_mobile,
                    has_touch=is_mobile,
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=7000)
                except PlaywrightError:
                    pass
                await page.screenshot(path=str(screenshot_path), full_page=True)
                return
            except PlaywrightError as exc:
                last_error = exc
            finally:
                if browser:
                    await browser.close()

    raise RuntimeError(
        "Could not open the website in Chrome or Edge. Install Google Chrome or Microsoft Edge, then try again."
    ) from last_error


def _screenshot_to_video(screenshot_path: Path, video_path: Path) -> None:
    ensure_local_ffmpeg_on_path()
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(screenshot_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        "6",
        "-vf",
        f"scale={VIDEO_SIZE}:force_original_aspect_ratio=decrease,pad={VIDEO_SIZE}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r",
        "24",
        "-shortest",
        str(video_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is not available yet. Run Start_TRIBE_Review.cmd again to finish setup.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
        raise RuntimeError(f"Could not prepare the website screenshot for analysis: {detail[0]}") from exc
