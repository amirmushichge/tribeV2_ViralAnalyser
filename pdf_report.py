from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import subprocess
import tempfile
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


plt.rcParams["font.family"] = "DejaVu Sans"

INK = "#102031"
MUTED = "#5f7182"
BLUE = "#2f8cff"
GREEN = "#38b26c"
CARD = "#f4f7fb"
CARD_EDGE = "#dbe4ee"
PANEL = "#eef6f8"


def render_html_pdf(html: str, timeout_seconds: int = 60) -> bytes:
    chrome_path = _find_chrome_executable()
    if not chrome_path:
        raise RuntimeError("Chrome is required to render the HTML report PDF, but it was not found.")

    with tempfile.TemporaryDirectory(prefix="tribe_pdf_") as temp_dir:
        temp_path = Path(temp_dir)
        html_path = temp_path / "report.html"
        pdf_path = temp_path / "report.pdf"
        html_path.write_text(html, encoding="utf-8")

        command = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                timeout=timeout_seconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"Chrome failed to render PDF. {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Chrome timed out while rendering the PDF report.") from exc

        if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
            raise RuntimeError("Chrome did not create a PDF report.")
        return pdf_path.read_bytes()


def _find_chrome_executable() -> str | None:
    env_value = os.environ.get("TRIBE_CHROME_PATH")
    candidates = [
        env_value,
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def render_pdf_report(report: dict) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for page in _build_pages(report):
            pdf.savefig(page, bbox_inches="tight")
            plt.close(page)
    return buffer.getvalue()


def _build_pages(report: dict) -> list:
    if _is_compare_report(report):
        return [_compare_summary_page(report), _compare_recommendations_page(report)]
    pages = [_summary_page(report), _recommendations_page(report)]
    return pages


def _is_compare_report(report: dict) -> bool:
    return report.get("mode") == "compare" or bool(report.get("variants"))


def _compare_summary_page(report: dict):
    language = report.get("report_language", "ru")
    fig = _page()
    gs = fig.add_gridspec(18, 6, left=0.06, right=0.94, top=0.96, bottom=0.05, hspace=0.55, wspace=0.42)
    ax_header = fig.add_subplot(gs[0:4, :])
    ax_timeline = fig.add_subplot(gs[4:10, :])
    ax_rank = fig.add_subplot(gs[10:15, :])
    ax_recs = fig.add_subplot(gs[15:18, :])
    for ax in [ax_header, ax_rank, ax_recs]:
        _hide_axis(ax)

    title = "Comparison of versions" if language == "en" else "Сравнение версий"
    subtitle = (
        "The report overlays the score curves and ranks the uploaded cuts."
        if language == "en"
        else "Отчёт накладывает графики роликов и ранжирует загруженные версии."
    )
    ax_header.text(0.0, 0.92, "TRIBE v2 Video Review", fontsize=25, fontweight="bold", color=INK, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.75, _report_stamp(report), fontsize=9.5, color=MUTED, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.49, title, fontsize=17, fontweight="bold", color=INK, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.29, "\n".join(wrap(subtitle, width=92)), fontsize=11, color="#3e5162", linespacing=1.35, transform=ax_header.transAxes)
    verdict = str(report.get("verdict") or "")
    summary = str(report.get("executive_summary") or "")
    if verdict or summary:
        _callout(ax_header, 0.0, 0.02, 0.98, 0.17, " ".join(item for item in [verdict, summary] if item))
    elif report.get("product_summary"):
        _callout(ax_header, 0.0, 0.02, 0.98, 0.17, str(report.get("product_summary")))

    _compare_timeline_chart(ax_timeline, report, language)

    ax_rank.text(0.0, 0.95, "Ranking" if language == "en" else "Рейтинг версий", fontsize=14, fontweight="bold", color=INK, transform=ax_rank.transAxes)
    ranking = [item for item in report.get("ranking", []) if isinstance(item, dict)]
    for index, item in enumerate(ranking[:4]):
        x = 0.01 + (index % 2) * 0.49
        y = 0.58 - (index // 2) * 0.32
        label = f"#{item.get('rank', index + 1)} - {item.get('name', '-')}"
        value = str(item.get("overall_score", "-"))
        _info_card(ax_rank, x, y, 0.46, 0.22, label, value)
        ax_rank.text(x + 0.02, y - 0.06, _wrap_pdf(item.get("summary") or "", width=48), fontsize=8.8, color="#405365", transform=ax_rank.transAxes)

    ax_recs.text(0.0, 0.95, "Next checks" if language == "en" else "Что проверить дальше", fontsize=14, fontweight="bold", color=INK, transform=ax_recs.transAxes)
    recs = [str(item) for item in report.get("recommendations", []) if str(item).strip()]
    if not recs:
        recs = ["Compare the strongest and weakest sections on the overlaid curve." if language == "en" else "Сравните сильные и слабые участки на наложенном графике."]
    for index, item in enumerate(recs[:3], start=1):
        ax_recs.text(0.02, 0.78 - (index - 1) * 0.22, f"{index}. " + _wrap_pdf(item, width=98), fontsize=9.8, color="#334355", linespacing=1.35, transform=ax_recs.transAxes)
    return fig


def _compare_recommendations_page(report: dict):
    language = report.get("report_language", "ru")
    fig = _page()
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    _hide_axis(ax)

    title = "Comparison details" if language == "en" else "Детали сравнения"
    hint = (
        "Use this page to decide which cut should become the base version for the next edit."
        if language == "en"
        else "Используйте эту страницу, чтобы выбрать базовую версию для следующей правки."
    )
    ax.text(0.0, 0.98, title, fontsize=20, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    _callout(ax, 0.0, 0.87, 0.98, 0.08, hint)

    y = 0.79
    recommendations = [str(item) for item in report.get("recommendations", []) if str(item).strip()]
    if recommendations:
        ax.text(0.0, y, "What to do next" if language == "en" else "Что делать дальше", fontsize=14, fontweight="bold", color=INK, transform=ax.transAxes)
        y -= 0.07
        for index, item in enumerate(recommendations[:5], start=1):
            ax.text(0.02, y, f"{index}. " + _wrap_pdf(item, width=100), fontsize=10, color="#334355", linespacing=1.35, transform=ax.transAxes)
            y -= 0.105

    rows = [item for item in report.get("comparison_rows", []) if isinstance(item, dict)]
    if rows:
        y = min(y - 0.02, 0.37)
        ax.text(0.0, y, "Metric winners" if language == "en" else "Победители по метрикам", fontsize=14, fontweight="bold", color=INK, transform=ax.transAxes)
        y -= 0.07
        for row in rows[:6]:
            label = str(row.get("label") or row.get("key") or "")
            winner = str(row.get("winner_name") or "-")
            score = str(row.get("winner_score") or "-")
            spread = str(row.get("spread") or "0")
            text = (
                f"{label}: {winner} leads with {score}; spread {spread}."
                if language == "en"
                else f"{label}: лидирует {winner} с {score}; разрыв {spread}."
            )
            ax.text(0.02, y, _wrap_pdf(text, width=102), fontsize=9.8, color="#334355", transform=ax.transAxes)
            y -= 0.065

    gaps = [str(item) for item in report.get("common_gaps", []) if str(item).strip()]
    if gaps:
        y = min(y - 0.03, 0.18)
        ax.text(0.0, y, "Shared weak points" if language == "en" else "Общие слабые места", fontsize=14, fontweight="bold", color=INK, transform=ax.transAxes)
        y -= 0.065
        for item in gaps[:3]:
            ax.text(0.02, y, "- " + _wrap_pdf(item, width=100), fontsize=9.8, color="#334355", transform=ax.transAxes)
            y -= 0.065
    return fig


def _summary_page(report: dict):
    language = report.get("report_language", "ru")
    fig = _page()
    gs = fig.add_gridspec(18, 6, left=0.06, right=0.94, top=0.96, bottom=0.05, hspace=0.62, wspace=0.45)
    ax_header = fig.add_subplot(gs[0:4, :])
    ax_cards = fig.add_subplot(gs[4:7, :])
    ax_timeline = fig.add_subplot(gs[7:13, :])
    ax_notes = fig.add_subplot(gs[13:18, :])
    for ax in [ax_header, ax_cards, ax_notes]:
        _hide_axis(ax)

    title = "Video score curve" if language == "en" else "График ролика"
    subtitle = (
        "Official TRIBE v2 inference output for one uploaded video."
        if language == "en"
        else "Официальный inference output TRIBE v2 для одного загруженного видео."
    )
    banner = (
        "The main guide is the curve. Compare dips with stronger moments and test edits against each other."
        if language == "en"
        else "Главный ориентир отчёта - график ролика. Сравнивайте провалы с более сильными участками и проверяйте правки сравнительными тестами."
    )

    ax_header.text(0.0, 0.92, "TRIBE v2 Video Review", fontsize=25, fontweight="bold", color=INK, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.75, _report_stamp(report), fontsize=9.5, color=MUTED, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.49, title, fontsize=16, fontweight="bold", color=INK, transform=ax_header.transAxes)
    ax_header.text(0.0, 0.27, "\n".join(wrap(subtitle, width=92)), fontsize=11, color="#3e5162", linespacing=1.35, transform=ax_header.transAxes)
    _callout(ax_header, 0.0, 0.02, 0.98, 0.17, banner)

    meta_cards = [
        ("duration" if language == "en" else "длительность", f"{_video(report).get('duration_seconds', '-')} s"),
        ("video size" if language == "en" else "размер видео", str(_video(report).get("resolution", "-"))),
        ("events" if language == "en" else "частей в разборе", str(_predictions(report).get("events_count", "-"))),
        ("best moment" if language == "en" else "самый сильный момент", str(_predictions(report).get("peak_time", "-"))),
    ]
    for idx, (label, value) in enumerate(meta_cards):
        _info_card(ax_cards, 0.01 + idx * 0.245, 0.27, 0.22, 0.42, label, value)

    _timeline_chart(ax_timeline, report, language)

    notes_title = "How to read it" if language == "en" else "Как читать отчёт"
    notes = (
        [
            "TRIBE v2 maps video, audio, and text into a model-based comparison space.",
            "The curve is a model-based read of the cut, not a measurement from one person.",
            "Use low points as edit candidates, but compare them with nearby high points before changing the cut.",
        ]
        if language == "en"
        else [
            "TRIBE v2 сопоставляет видео, звук и текст с модельным пространством реакции мозга.",
            "Это модель усреднённого зрителя, а не измерение реакции одного конкретного человека.",
            "Провалы используйте как кандидаты на правку, но сначала сравнивайте их с соседними сильными участками.",
        ]
    )
    ax_notes.text(0.0, 0.95, notes_title, fontsize=14, fontweight="bold", color=INK, transform=ax_notes.transAxes)
    for idx, item in enumerate(notes, start=1):
        ax_notes.text(0.02, 0.82 - (idx - 1) * 0.18, f"{idx}. " + "\n".join(wrap(item, width=98)), fontsize=10.5, color="#334355", linespacing=1.42, transform=ax_notes.transAxes)
    return fig


def _recommendations_page(report: dict):
    language = report.get("report_language", "ru")
    fig = _page()
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    _hide_axis(ax)

    title = "Recommendations" if language == "en" else "Рекомендации и точки внимания"
    hint = (
        "The cards below are general recommendations. The key place to look is the curve: find dips, compare them with stronger moments, experiment with edits, and run comparative tests."
        if language == "en"
        else "Ниже приведены общие рекомендации. Главное - смотреть на график ролика: находите провалы, сравнивайте их с более сильными местами, экспериментируйте с правками и проводите сравнительные тесты."
    )
    ax.text(0.0, 0.98, title, fontsize=20, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    _callout(ax, 0.0, 0.87, 0.98, 0.08, hint)

    y = 0.77
    action_items = _action_items(report)
    if not action_items:
        empty = "No editorial recommendations were generated." if language == "en" else "Редакторские рекомендации не были сформированы."
        ax.text(0.02, y, empty, fontsize=11, color=MUTED, transform=ax.transAxes)
        y -= 0.1

    for item in action_items[:6]:
        y = _recommendation_strip(ax, y, item)

    metrics = _metrics(report)
    if metrics:
        ax.text(0.0, max(y - 0.01, 0.22), "Metrics" if language == "en" else "Ключевые метрики", fontsize=14, fontweight="bold", color=INK, transform=ax.transAxes)
        y = max(y - 0.09, 0.15)
        for metric in metrics[:4]:
            y = _metric_row(ax, y, metric)
    return fig


def _details_page(report: dict):
    language = report.get("report_language", "ru")
    fig = _page()
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    _hide_axis(ax)

    ax.text(0.0, 0.98, "Technical details" if language == "en" else "Технические детали", fontsize=18, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.90, "TRIBE v2 workflow", fontsize=13, fontweight="bold", color="#18324b", transform=ax.transAxes)
    code = (
        'from tribev2 import TribeModel\n\n'
        'model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")\n'
        'df = model.get_events_dataframe(video_path="path/to/video.mp4")\n'
        'preds, segments = model.predict(events=df)'
    )
    ax.text(0.0, 0.79, code, fontsize=10.2, color="#334355", family="monospace", linespacing=1.55, transform=ax.transAxes)

    ax.text(0.0, 0.57, "Run metadata" if language == "en" else "Метаданные прогона", fontsize=13, fontweight="bold", color="#18324b", transform=ax.transAxes)
    metadata = [
        f"filename: {_video(report).get('filename', '-')}",
        f"duration_seconds: {_video(report).get('duration_seconds', '-')}",
        f"fps: {_video(report).get('fps', '-')}",
        f"resolution: {_video(report).get('resolution', '-')}",
        f"device: {report.get('device', '-')}",
        f"events_count: {_predictions(report).get('events_count', '-')}",
        f"peak_time: {_predictions(report).get('peak_time', '-')}",
    ]
    for idx, item in enumerate(metadata):
        ax.text(0.02, 0.51 - idx * 0.055, item, fontsize=10.5, color="#334355", transform=ax.transAxes)

    sources = report.get("official_sources") or {}
    if isinstance(sources, dict) and sources:
        ax.text(0.0, 0.09, "Sources: " + " | ".join(str(value) for value in sources.values()), fontsize=8.8, color=MUTED, transform=ax.transAxes)
    return fig


def _timeline_chart(ax, report: dict, language: str) -> None:
    points = ((report.get("timeline") or {}).get("points") or [])
    xs = [point.get("seconds", 0) for point in points if isinstance(point, dict)]
    ys = [point.get("signal_score", 0) for point in points if isinstance(point, dict)]
    if not xs or not ys:
        ax.text(0.02, 0.5, "No timeline data" if language == "en" else "Нет данных графика", fontsize=11, color=MUTED, transform=ax.transAxes)
        _hide_axis(ax)
        return

    ax.plot(xs, ys, color=BLUE, linewidth=2.7)
    ax.fill_between(xs, ys, 0, color=GREEN, alpha=0.12)
    ax.set_title("Curve over time" if language == "en" else "График по времени", loc="left", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylabel("Level" if language == "en" else "Уровень", color=MUTED)
    ax.set_xlabel("Seconds" if language == "en" else "Секунды", color=MUTED)
    ax.grid(alpha=0.22, linestyle="--")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#cfd9e4")


def _compare_timeline_chart(ax, report: dict, language: str) -> None:
    overlay = report.get("timeline_overlay") if isinstance(report.get("timeline_overlay"), dict) else {}
    series = [item for item in overlay.get("series", []) if isinstance(item, dict)]
    palette = ["#2f8cff", "#38b26c", "#f0a33a", "#b765ff"]
    plotted = 0
    max_seconds = float(overlay.get("duration_seconds") or 0.0)

    for index, variant in enumerate(report.get("variants", [])):
        if not isinstance(variant, dict):
            continue
        matching = next((item for item in series if item.get("variant_key") == variant.get("variant_key")), {})
        color = str(matching.get("color") or palette[index % len(palette)])
        points = ((variant.get("timeline") or {}).get("points") or [])
        xs = [point.get("seconds", 0) for point in points if isinstance(point, dict)]
        ys = [point.get("signal_score", 0) for point in points if isinstance(point, dict)]
        if xs and ys:
            ax.plot(xs, ys, color=color, linewidth=2.4, label=_pdf_text(variant.get("title") or variant.get("variant_key") or "Version"))
            plotted += 1
            max_seconds = max(max_seconds, max(float(value or 0) for value in xs))

    if not plotted:
        ax.text(0.02, 0.5, "No timeline data" if language == "en" else "Нет данных графика", fontsize=11, color=MUTED, transform=ax.transAxes)
        _hide_axis(ax)
        return

    ax.set_title("Overlaid curves" if language == "en" else "Наложение графиков", loc="left", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylim(0, 100)
    duration = max(max_seconds, 1.0)
    ax.set_xlim(0, max(duration, 1.0))
    ax.set_ylabel("Level" if language == "en" else "Уровень", color=MUTED)
    ax.set_xlabel("Seconds" if language == "en" else "Секунды", color=MUTED)
    ax.grid(alpha=0.22, linestyle="--")
    ax.tick_params(colors=MUTED, labelsize=9)
    if plotted:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#cfd9e4")


def _recommendation_strip(ax, y: float, item: dict) -> float:
    if y < 0.12:
        return y
    _rounded_box(ax, 0.0, y - 0.105, 0.98, 0.09, fc="#f8fbff")
    timestamp = str(item.get("timestamp") or item.get("time") or "")
    title = str(item.get("title") or item.get("label") or "Recommendation")
    text = str(item.get("instruction") or item.get("description") or item.get("reason") or "")
    ax.text(0.025, y - 0.038, timestamp, fontsize=10.5, color=BLUE, fontweight="bold", transform=ax.transAxes)
    ax.text(0.14, y - 0.038, _pdf_text(title), fontsize=10.8, color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.14, y - 0.072, _wrap_pdf(text, width=104), fontsize=9.4, color="#405365", transform=ax.transAxes)
    return y - 0.12


def _metric_row(ax, y: float, metric: dict) -> float:
    if y < 0.08:
        return y
    score = metric.get("score", "-")
    label = str(metric.get("label") or metric.get("name") or "")
    summary = str(metric.get("summary") or metric.get("description") or "")
    _rounded_box(ax, 0.0, y - 0.075, 0.98, 0.06, fc="#fbfcfe")
    ax.text(0.025, y - 0.052, str(score), fontsize=13, color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.11, y - 0.04, _pdf_text(label), fontsize=10.2, color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.33, y - 0.04, _wrap_pdf(summary, width=70), fontsize=9.1, color="#405365", transform=ax.transAxes)
    return y - 0.085


def _page():
    return plt.figure(figsize=(8.27, 11.69), facecolor="white")


def _hide_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _rounded_box(ax, x, y, w, h, fc=CARD, ec=CARD_EDGE) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=1, edgecolor=ec, facecolor=fc, transform=ax.transAxes)
    ax.add_patch(patch)


def _callout(ax, x, y, w, h, text: str) -> None:
    _rounded_box(ax, x, y, w, h, fc=PANEL, ec="#c7e3e6")
    ax.text(x + 0.02, y + h - 0.035, _wrap_pdf(text, width=112), fontsize=9.5, color="#36505a", linespacing=1.25, va="top", transform=ax.transAxes)


def _info_card(ax, x, y, w, h, label, value) -> None:
    _rounded_box(ax, x, y, w, h)
    ax.text(x + 0.02, y + h - 0.11, _pdf_text(label), fontsize=9.2, color=MUTED, transform=ax.transAxes)
    ax.text(x + 0.02, y + 0.12, _pdf_text(value), fontsize=14.5, fontweight="bold", color=INK, transform=ax.transAxes)


def _video(report: dict) -> dict:
    return report.get("video") if isinstance(report.get("video"), dict) else {}


def _predictions(report: dict) -> dict:
    return report.get("predictions") if isinstance(report.get("predictions"), dict) else {}


def _action_items(report: dict) -> list[dict]:
    editorial = report.get("editorial") if isinstance(report.get("editorial"), dict) else {}
    items = editorial.get("action_items") if isinstance(editorial.get("action_items"), list) else []
    return [item for item in items if isinstance(item, dict)]


def _metrics(report: dict) -> list[dict]:
    editorial = report.get("editorial") if isinstance(report.get("editorial"), dict) else {}
    metrics = editorial.get("metrics") if isinstance(editorial.get("metrics"), list) else []
    return [metric for metric in metrics if isinstance(metric, dict)]


def _pdf_text(value) -> str:
    text = str(value)
    return "".join(char for char in text if ord(char) <= 0xFFFF and ord(char) != 0xFE0F)


def _wrap_pdf(value, width: int) -> str:
    return "\n".join(wrap(_pdf_text(value), width=width))


def _report_stamp(report: dict) -> str:
    report_id = report.get("report_id", "-")
    created_at = report.get("created_at", "-")
    return f"Report {report_id}  |  {created_at}"
