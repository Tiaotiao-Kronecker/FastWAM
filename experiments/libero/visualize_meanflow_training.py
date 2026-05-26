#!/usr/bin/env python3
"""Build a local MeanFlow training dashboard from FastWAM training logs.

The script has no third-party dependencies. It parses Rich-wrapped trainer logs,
writes a metrics CSV/JSON file, and renders a standalone HTML dashboard with
inline SVG charts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("/DATA/disk3/tmp/fastwam_meanflow_20260517")
DEFAULT_TRAIN_ROOT = Path("runs/libero_one_step_meanflow_2cam224_1e-4")
DEFAULT_OUTPUT_DIR = Path("evaluate_results/training/meanflow_20260517")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(
    r"epoch=(?P<epoch>\d+).*?step=(?P<step>\d+)/(?:\s*)?(?P<max_steps>\d+)"
)
KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_/]*)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
SPEED_RE = re.compile(
    r"speed=(?P<step_s>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) step/s,"
    r"\s*(?P<sample_s>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) samples/s"
)
ETA_RE = re.compile(r"eta=(?P<eta>\d{2}:\d{2}:\d{2})")
ERROR_HINTS = (
    "Error executing job",
    "RuntimeError:",
    "ValueError:",
    "ChildFailedError",
    "CUDA out of memory",
    "Traceback (most recent call last)",
)
METRIC_KEYS = {
    "loss",
    "loss_action_endpoint",
    "loss_action_velocity",
    "loss_action_endpoint_sanity",
    "loss_meanflow_target",
    "loss_meanflow_action",
    "loss_meanflow_video",
    "loss_video_endpoint_sanity",
    "meanflow_interval",
    "meanflow_sigma_end",
    "meanflow_sigma_start",
    "lr",
    "val_loss",
    "infer_psnr",
    "infer_ssim",
    "psnr_rg",
    "ssim_rg",
    "psnr_rd",
    "ssim_rd",
    "psnr_dg",
    "ssim_dg",
    "action_l1",
    "action_l2",
}


@dataclass
class RunMetrics:
    label: str
    log_path: Path
    points: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    max_steps_reached: bool = False


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_log_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty run label in {value!r}.")
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def _discover_logs(log_dir: Path, pattern: str) -> list[tuple[str, Path]]:
    if not log_dir.exists():
        return []
    specs = []
    for path in sorted(log_dir.glob(pattern)):
        if path.is_file():
            specs.append((path.stem, path))
    return specs


def _merge_key_values(row: dict[str, Any], line: str) -> None:
    speed_match = SPEED_RE.search(line)
    if speed_match:
        row["speed_step_s"] = float(speed_match.group("step_s"))
        row["speed_samples_s"] = float(speed_match.group("sample_s"))

    eta_match = ETA_RE.search(line)
    if eta_match:
        row["eta"] = eta_match.group("eta")

    for match in KEY_VALUE_RE.finditer(line):
        key = match.group("key")
        value = match.group("value")
        if key in {"epoch", "step", "speed", "eta"}:
            continue
        if key not in METRIC_KEYS:
            continue
        if key == "loss":
            row["loss"] = float(value)
        else:
            row[key.replace("/", "_")] = float(value)


def parse_training_log(label: str, log_path: Path) -> RunMetrics:
    metrics = RunMetrics(label=label, log_path=log_path)
    if not log_path.exists():
        metrics.errors.append(f"missing log file: {log_path}")
        return metrics

    current: dict[str, Any] | None = None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        metrics.errors.append(f"failed to read log file: {exc}")
        return metrics

    for raw_line in lines:
        line = _strip_ansi(raw_line).strip()
        if not line:
            continue

        step_match = STEP_RE.search(line)
        if step_match:
            if current is not None:
                metrics.points.append(current)
            current = {
                "run": label,
                "log_path": str(log_path),
                "epoch": int(step_match.group("epoch")),
                "step": int(step_match.group("step")),
                "max_steps": int(step_match.group("max_steps")),
            }
            _merge_key_values(current, line)
            continue

        if current is not None:
            _merge_key_values(current, line)

        if "max_steps reached" in line:
            metrics.max_steps_reached = True

        for hint in ERROR_HINTS:
            if hint in line:
                if len(metrics.errors) < 12:
                    metrics.errors.append(line)
                break

    if current is not None:
        metrics.points.append(current)

    # Keep the last parsed value for duplicate steps. Rich/elastic logs can
    # occasionally duplicate a block after failures.
    deduped: dict[int, dict[str, Any]] = {}
    for point in metrics.points:
        deduped[int(point["step"])] = point
    metrics.points = [deduped[step] for step in sorted(deduped)]
    return metrics


def _latest_checkpoint(train_root: Path, label: str) -> dict[str, str | int | None]:
    run_dir = train_root / label
    weights_dir = run_dir / "checkpoints" / "weights"
    state_dir = run_dir / "checkpoints" / "state"
    result: dict[str, str | int | None] = {
        "run_dir": str(run_dir),
        "latest_weight": None,
        "latest_weight_step": None,
        "latest_state": None,
        "latest_state_step": None,
    }
    if weights_dir.exists():
        weights = sorted(weights_dir.glob("step_*.pt"))
        if weights:
            latest = weights[-1]
            result["latest_weight"] = str(latest)
            result["latest_weight_step"] = _step_from_name(latest.name)
    if state_dir.exists():
        states = sorted(path for path in state_dir.glob("step_*") if path.is_dir())
        if states:
            latest = states[-1]
            result["latest_state"] = str(latest)
            result["latest_state_step"] = _step_from_name(latest.name)
    return result


def _step_from_name(name: str) -> int | None:
    match = re.search(r"step[_-](\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _all_metric_keys(runs: list[RunMetrics]) -> list[str]:
    fixed = ["run", "step", "max_steps", "epoch"]
    keys: set[str] = set()
    for run in runs:
        for point in run.points:
            keys.update(point.keys())
    keys.difference_update(fixed)
    keys.discard("log_path")
    ordered = fixed + sorted(keys)
    return ordered


def write_metrics_csv(runs: list[RunMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _all_metric_keys(runs)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            for point in run.points:
                writer.writerow(point)


def write_summary_json(runs: list[RunMetrics], train_root: Path, output_path: Path) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        latest = run.points[-1] if run.points else {}
        ckpt = _latest_checkpoint(train_root, run.label)
        status = "missing_log" if run.errors and not run.points else "running"
        if run.max_steps_reached:
            status = "complete"
        if run.errors and any("RuntimeError:" in err or "ValueError:" in err or "CUDA out of memory" in err for err in run.errors):
            status = "failed_or_interrupted"
        rows.append(
            {
                "run": run.label,
                "status": status,
                "log_path": str(run.log_path),
                "num_points": len(run.points),
                "latest": latest,
                "errors": run.errors,
                "checkpoint": ckpt,
            }
        )
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _series_for_metric(runs: list[RunMetrics], metric: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        points = []
        for point in run.points:
            value = point.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                points.append((float(point["step"]), float(value)))
        if points:
            series[run.label] = points
    return series


def _nice_range(values: list[float], *, log_y: bool) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if log_y:
        positive = [value for value in values if value > 0]
        if not positive:
            return 1e-6, 1.0
        low = min(positive)
        high = max(positive)
        if low == high:
            low *= 0.8
            high *= 1.2
        return low, high
    low = min(values)
    high = max(values)
    if low == high:
        pad = abs(low) * 0.1 if low else 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _format_number(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def render_svg_chart(
    series: dict[str, list[tuple[float, float]]],
    title: str,
    y_label: str,
    *,
    log_y: bool = False,
) -> str:
    width = 920
    height = 360
    margin_left = 72
    margin_right = 24
    margin_top = 44
    margin_bottom = 52
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
    ]

    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return f"<section class=\"chart\"><h2>{html.escape(title)}</h2><p>No data yet.</p></section>"

    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    if min_x == max_x:
        min_x = 0.0
        max_x = max_x + 1.0

    values = [point[1] for point in all_points]
    min_y, max_y = _nice_range(values, log_y=log_y)
    if log_y:
        min_y = max(min_y, 1e-12)
        max_y = max(max_y, min_y * 1.01)
        log_min_y = math.log10(min_y)
        log_max_y = math.log10(max_y)

        def scale_y(value: float) -> float:
            value = max(value, 1e-12)
            frac = (math.log10(value) - log_min_y) / max(log_max_y - log_min_y, 1e-12)
            return margin_top + plot_h * (1.0 - frac)

    else:

        def scale_y(value: float) -> float:
            frac = (value - min_y) / max(max_y - min_y, 1e-12)
            return margin_top + plot_h * (1.0 - frac)

    def scale_x(value: float) -> float:
        frac = (value - min_x) / max(max_x - min_x, 1e-12)
        return margin_left + plot_w * frac

    x_ticks = [min_x + (max_x - min_x) * i / 4.0 for i in range(5)]
    if log_y:
        y_ticks = [10 ** (math.log10(min_y) + (math.log10(max_y) - math.log10(min_y)) * i / 4.0) for i in range(5)]
    else:
        y_ticks = [min_y + (max_y - min_y) * i / 4.0 for i in range(5)]

    parts = [
        '<section class="chart">',
        f"<h2>{html.escape(title)}</h2>",
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    for tick in y_ticks:
        y = scale_y(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid"/>')
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">'
            f"{html.escape(_format_number(tick))}</text>"
        )
    for tick in x_ticks:
        x = scale_x(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" class="grid"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{height - 20}" text-anchor="middle" class="tick">'
            f"{int(round(tick))}</text>"
        )

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis"/>')
    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>'
    )
    parts.append(f'<text x="{width / 2:.2f}" y="{height - 5}" text-anchor="middle" class="axis-label">step</text>')
    parts.append(
        f'<text x="18" y="{height / 2:.2f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {height / 2:.2f})">{html.escape(y_label)}</text>'
    )

    for idx, (label, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        polyline = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points[-12:]:
            parts.append(f'<circle cx="{scale_x(x):.2f}" cy="{scale_y(y):.2f}" r="3" fill="{color}"/>')

    legend_x = margin_left
    legend_y = 18
    for idx, label in enumerate(series.keys()):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 18
        parts.append(f'<rect x="{legend_x}" y="{y - 9}" width="11" height="11" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 16}" y="{y}" class="legend">{html.escape(label)}</text>')

    parts.append("</svg>")
    parts.append("</section>")
    return "\n".join(parts)


def _html_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "status",
        "step",
        "max_steps",
        "loss",
        "loss_meanflow_target",
        "loss_meanflow_action",
        "loss_meanflow_video",
        "speed_step_s",
        "eta",
        "latest_weight",
    ]
    body = []
    for row in rows:
        latest = row.get("latest", {})
        ckpt = row.get("checkpoint", {})
        values = {
            "run": row.get("run"),
            "status": row.get("status"),
            "step": latest.get("step"),
            "max_steps": latest.get("max_steps"),
            "loss": latest.get("loss"),
            "loss_meanflow_target": latest.get("loss_meanflow_target"),
            "loss_meanflow_action": latest.get("loss_meanflow_action"),
            "loss_meanflow_video": latest.get("loss_meanflow_video"),
            "speed_step_s": latest.get("speed_step_s"),
            "eta": latest.get("eta"),
            "latest_weight": ckpt.get("latest_weight"),
        }
        cells = []
        for header in headers:
            value = values.get(header)
            if isinstance(value, float):
                text = _format_number(value)
            elif value is None:
                text = ""
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return "<table><thead><tr>" + header_html + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"


def write_html_dashboard(
    *,
    runs: list[RunMetrics],
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    csv_path: Path,
    json_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    charts = [
        render_svg_chart(_series_for_metric(runs, "loss"), "Total Loss", "loss", log_y=True),
        render_svg_chart(
            _series_for_metric(runs, "loss_meanflow_target"),
            "MeanFlow Target Loss",
            "loss_meanflow_target",
            log_y=True,
        ),
        render_svg_chart(
            _series_for_metric(runs, "loss_meanflow_action"),
            "MeanFlow Action Loss",
            "loss_meanflow_action",
            log_y=True,
        ),
        render_svg_chart(
            _series_for_metric(runs, "loss_meanflow_video"),
            "MeanFlow Video Loss",
            "loss_meanflow_video",
            log_y=True,
        ),
        render_svg_chart(
            _series_for_metric(runs, "loss_action_endpoint_sanity"),
            "Action Endpoint Sanity Loss",
            "loss_action_endpoint_sanity",
            log_y=True,
        ),
        render_svg_chart(
            _series_for_metric(runs, "loss_video_endpoint_sanity"),
            "Video Endpoint Sanity Loss",
            "loss_video_endpoint_sanity",
            log_y=True,
        ),
        render_svg_chart(_series_for_metric(runs, "speed_step_s"), "Training Speed", "step/s"),
        render_svg_chart(_series_for_metric(runs, "lr"), "Learning Rate", "lr", log_y=True),
        render_svg_chart(_series_for_metric(runs, "meanflow_interval"), "MeanFlow Interval", "sigma_end - sigma_start"),
        render_svg_chart(_series_for_metric(runs, "meanflow_sigma_start"), "MeanFlow Sigma Start", "sigma_start"),
        render_svg_chart(_series_for_metric(runs, "meanflow_sigma_end"), "MeanFlow Sigma End", "sigma_end"),
    ]

    error_blocks = []
    for row in summary_rows:
        errors = row.get("errors") or []
        if errors:
            error_blocks.append(
                f"<h3>{html.escape(row['run'])}</h3><pre>{html.escape(chr(10).join(errors))}</pre>"
            )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f5;
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 6px;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 14px;
    }}
    h3 {{
      font-size: 15px;
      margin: 16px 0 6px;
    }}
    .muted {{
      color: #667085;
      margin-top: 0;
    }}
    .panel, .chart {{
      background: #fff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f2f4f7;
      font-weight: 650;
    }}
    code {{
      background: #eef2f6;
      border-radius: 4px;
      padding: 1px 5px;
    }}
    pre {{
      background: #111827;
      color: #f9fafb;
      padding: 12px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 12px;
    }}
    svg {{
      width: 100%;
      min-width: 760px;
      height: auto;
    }}
    .grid {{
      stroke: #e5e7eb;
      stroke-width: 1;
    }}
    .axis {{
      stroke: #111827;
      stroke-width: 1.2;
    }}
    .tick, .legend, .axis-label {{
      fill: #4b5563;
      font-size: 12px;
    }}
    .legend {{
      font-weight: 600;
    }}
    ul {{
      margin-top: 8px;
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="muted">Generated from FastWAM training logs. Re-run the script to refresh while jobs are running.</p>

  <section class="panel">
    <h2>Run Summary</h2>
    {_html_table(summary_rows)}
    <p class="muted">CSV: <code>{html.escape(str(csv_path))}</code> JSON: <code>{html.escape(str(json_path))}</code></p>
  </section>

  <section class="panel">
    <h2>How To Read</h2>
    <ul>
      <li><code>Total Loss</code> and <code>MeanFlow Target Loss</code> should trend down. Occasional spikes are normal because r/t are sampled randomly.</li>
      <li><code>Training Speed</code> should be stable after warmup. A sudden long drop usually means GPU contention, dataloader stalls, or a hung worker.</li>
      <li><code>Learning Rate</code> confirms whether a run was resumed from full state or restarted from weights.</li>
      <li><code>MeanFlow Interval</code>, <code>Sigma Start</code>, and <code>Sigma End</code> should stay spread out for random r/t. If interval collapses near zero, the training target is not exercising the intended range.</li>
      <li>For this project, loss is only a gate. The final decision still depends on gap-probe rollout success and gripper/action-trace timing.</li>
    </ul>
  </section>

  {''.join(charts)}

  <section class="panel">
    <h2>Errors</h2>
    {''.join(error_blocks) if error_blocks else '<p>No error hints found in parsed logs.</p>'}
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Run log as LABEL=PATH or PATH. Can be passed multiple times.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="MeanFlow Training Dashboard")
    args = parser.parse_args()

    log_specs = [_parse_log_spec(value) for value in args.log]
    if not log_specs:
        log_specs = _discover_logs(args.log_dir, args.pattern)
    if not log_specs:
        raise SystemExit(
            f"No logs found. Pass --log LABEL=PATH or create logs under {args.log_dir}."
        )

    runs = [parse_training_log(label, path) for label, path in log_specs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training_metrics.csv"
    json_path = args.output_dir / "training_summary.json"
    html_path = args.output_dir / "index.html"

    write_metrics_csv(runs, csv_path)
    summary_rows = write_summary_json(runs, args.train_root, json_path)
    write_html_dashboard(
        runs=runs,
        summary_rows=summary_rows,
        output_path=html_path,
        title=args.title,
        csv_path=csv_path,
        json_path=json_path,
    )

    print(f"Wrote dashboard: {html_path}")
    print(f"Wrote metrics CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")


if __name__ == "__main__":
    main()
