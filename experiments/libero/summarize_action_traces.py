#!/usr/bin/env python3
"""Summarize LIBERO action traces with gripper/timing diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TRACE_RE = re.compile(r"task(?P<task_id>\d+)_trial(?P<trial>\d+)_success(?P<success>True|False)_action_trace\.json$")


def _parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Run label is empty in {value!r}.")
        return label, Path(path)
    path = Path(value)
    return path.name, path


def _iter_trace_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/action_traces/*_action_trace.json"))


def _suite_from_trace_path(trace_path: Path) -> str:
    # <run>/<suite>/action_traces/<trace>.json
    try:
        return trace_path.parents[1].name
    except IndexError:
        return "unknown"


def _load_stage_actions(trace: dict[str, Any], stage: str) -> list[list[float]]:
    if stage == "executed_policy_actions":
        return [record["action"] for record in trace.get("executed_policy_actions", [])]

    actions = []
    for record in trace.get("executed_policy_actions", []):
        action_by_stage = record.get("action_by_stage", {})
        if stage in action_by_stage:
            actions.append(action_by_stage[stage])
    if actions:
        return actions

    for replan in trace.get("replans", []):
        stage_actions = replan.get("chunk_actions_by_stage", {}).get(stage)
        if stage_actions:
            executed_count = int(replan.get("executed_count", len(stage_actions)))
            actions.extend(stage_actions[:executed_count])
    return actions


def _segments(values: np.ndarray) -> list[tuple[int, int, float]]:
    if values.size == 0:
        return []
    segments = []
    start = 0
    current = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        value = float(value)
        if value != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = value
    segments.append((start, int(values.size - 1), current))
    return segments


def _format_segments(segments: list[tuple[int, int, float]], max_segments: int) -> str:
    shown = [f"{start}-{end}:{value:+g}" for start, end, value in segments[:max_segments]]
    if len(segments) > max_segments:
        shown.append(f"...({len(segments)} segments)")
    return "; ".join(shown)


def _summarize_actions(actions: list[list[float]], max_segments: int) -> dict[str, Any]:
    if not actions:
        return {
            "n_actions": 0,
            "first_close_step": None,
            "close_ratio": None,
            "gripper_transitions": 0,
            "gripper_segments": "",
            "max_pos": None,
            "max_rot": None,
            "pos_abs_ge_0_90": None,
            "rot_abs_ge_0_35": None,
            "gripper_min": None,
            "gripper_max": None,
            "gripper_mean": None,
        }

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Expected action array [T, >=7], got {arr.shape}.")

    gripper = arr[:, -1]
    gripper_sign = np.sign(gripper).astype(np.float32)
    close_mask = gripper_sign > 0
    first_close = int(np.argmax(close_mask)) if bool(np.any(close_mask)) else None
    segments = _segments(gripper_sign)
    max_pos = float(np.max(np.abs(arr[:, :3]))) if arr.shape[0] else None
    max_rot = float(np.max(np.abs(arr[:, 3:6]))) if arr.shape[0] else None

    return {
        "n_actions": int(arr.shape[0]),
        "first_close_step": first_close,
        "close_ratio": float(np.mean(close_mask)),
        "gripper_transitions": max(0, len(segments) - 1),
        "gripper_segments": _format_segments(segments, max_segments=max_segments),
        "max_pos": max_pos,
        "max_rot": max_rot,
        "pos_abs_ge_0_90": float(np.mean(np.abs(arr[:, :3]) >= 0.90)),
        "rot_abs_ge_0_35": float(np.mean(np.abs(arr[:, 3:6]) >= 0.35)),
        "gripper_min": float(np.min(gripper)),
        "gripper_max": float(np.max(gripper)),
        "gripper_mean": float(np.mean(gripper)),
    }


def _summarize_trace(
    *,
    run_label: str,
    trace_path: Path,
    stage: str,
    max_segments: int,
) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    match = TRACE_RE.search(trace_path.name)
    if match is None:
        task_id = None
        trial = None
        success_from_name = None
    else:
        task_id = int(match.group("task_id"))
        trial = int(match.group("trial"))
        success_from_name = match.group("success") == "True"

    actions = _load_stage_actions(trace, stage)
    summary = _summarize_actions(actions, max_segments=max_segments)
    summary.update(
        {
            "run": run_label,
            "suite": _suite_from_trace_path(trace_path),
            "task_id": task_id,
            "trial": trial,
            "success": bool(trace.get("success", success_from_name)),
            "stage": stage,
            "n_replans": int(len(trace.get("replans", []))),
            "task_description": trace.get("task_description", ""),
            "trace_path": str(trace_path),
        }
    )
    return summary


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path, stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"action_trace_summary_{stage}.json"
    csv_path = output_dir / f"action_trace_summary_{stage}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "run",
        "suite",
        "task_id",
        "trial",
        "success",
        "stage",
        "n_actions",
        "n_replans",
        "first_close_step",
        "close_ratio",
        "gripper_transitions",
        "gripper_segments",
        "max_pos",
        "max_rot",
        "pos_abs_ge_0_90",
        "rot_abs_ge_0_35",
        "gripper_min",
        "gripper_max",
        "gripper_mean",
        "task_description",
        "trace_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _print_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        ("run", 11),
        ("suite", 14),
        ("task_id", 4),
        ("success", 7),
        ("n_actions", 7),
        ("first_close_step", 10),
        ("close_ratio", 8),
        ("gripper_transitions", 5),
        ("max_pos", 7),
        ("max_rot", 7),
    ]
    header = " ".join(name[:width].ljust(width) for name, width in fields)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for name, width in fields:
            value = row.get(name)
            if isinstance(value, float):
                value = f"{value:.3f}"
            elif value is None:
                value = "-"
            else:
                value = str(value)
            values.append(value[:width].ljust(width))
        print(" ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory, optionally labeled as label=/path/to/run.",
    )
    parser.add_argument(
        "--stage",
        default="executed_policy_actions",
        help=(
            "Trace stage to summarize. Use executed_policy_actions for old traces, "
            "or a raw stage such as env_action/model_normalized for new traces."
        ),
    )
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for run_arg in args.run:
        label, run_dir = _parse_run_arg(run_arg)
        trace_files = _iter_trace_files(run_dir)
        if not trace_files:
            raise FileNotFoundError(f"No action traces found under {run_dir}.")
        for trace_path in trace_files:
            rows.append(
                _summarize_trace(
                    run_label=label,
                    trace_path=trace_path,
                    stage=args.stage,
                    max_segments=args.max_segments,
                )
            )

    rows.sort(key=lambda row: (str(row["suite"]), int(row["task_id"]), str(row["run"])))
    _print_table(rows)
    if args.output_dir is not None:
        _write_outputs(rows, args.output_dir, args.stage)
        print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
