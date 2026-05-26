#!/usr/bin/env python3
"""Replay original and hybrid LIBERO action traces for gripper/timing diagnosis."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("LIBERO_CONFIG_PATH", "/DATA/disk3/tmp/libero_config")

from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from libero.libero import benchmark  # noqa: E402


TRACE_RE = re.compile(r"task(?P<task_id>\d+)_trial(?P<trial>\d+)_success(?P<success>True|False)_action_trace\.json$")


VARIANTS = (
    "meanflow_original",
    "release_original",
    "meanflow_motion_release_gripper",
    "release_motion_meanflow_gripper",
    "meanflow_motion_smoothed_gripper",
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_task_file(path: Path) -> list[tuple[str, int]]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid task line in {path}: {line!r}")
        tasks.append((parts[0], int(parts[1])))
    if not tasks:
        raise ValueError(f"No tasks found in {path}.")
    return tasks


def _trace_path(run_dir: Path, suite: str, task_id: int, trial: int) -> Path:
    trace_dir = run_dir / suite / "action_traces"
    matches = sorted(trace_dir.glob(f"task{task_id}_trial{trial}_success*_action_trace.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one trace for {suite} task{task_id} trial{trial} under {trace_dir}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _load_trace_actions(run_dir: Path, suite: str, task_id: int, trial: int) -> tuple[dict[str, Any], np.ndarray]:
    path = _trace_path(run_dir, suite, task_id, trial)
    trace = json.loads(path.read_text(encoding="utf-8"))
    actions = np.asarray(
        [record["action"] for record in trace.get("executed_policy_actions", [])],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Invalid action trace shape in {path}: {actions.shape}.")
    trace["_trace_path"] = str(path)
    return trace, actions


def _pad_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.copy()
    if len(values) > length:
        return values[:length].copy()
    if len(values) == 0:
        raise ValueError("Cannot pad an empty gripper sequence.")
    pad = np.full((length - len(values),), values[-1], dtype=values.dtype)
    return np.concatenate([values, pad], axis=0)


def _segments(values: np.ndarray) -> list[tuple[int, int, float]]:
    if len(values) == 0:
        return []
    out = []
    start = 0
    current = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        value = float(value)
        if value != current:
            out.append((start, idx - 1, current))
            start = idx
            current = value
    out.append((start, len(values) - 1, current))
    return out


def _smooth_gripper(gripper: np.ndarray, min_run: int) -> np.ndarray:
    values = np.sign(gripper).astype(np.float32).copy()
    if len(values) == 0 or min_run <= 1:
        return values

    changed = True
    while changed:
        changed = False
        segments = _segments(values)
        if len(segments) <= 1:
            break
        for seg_idx, (start, end, value) in enumerate(segments):
            length = end - start + 1
            if length >= min_run:
                continue
            if seg_idx > 0:
                replacement = segments[seg_idx - 1][2]
            elif seg_idx + 1 < len(segments):
                replacement = segments[seg_idx + 1][2]
            else:
                replacement = value
            if replacement != value:
                values[start : end + 1] = replacement
                changed = True
                break
    return values


def _build_variant_actions(
    variant: str,
    meanflow_actions: np.ndarray,
    release_actions: np.ndarray,
    *,
    smooth_min_run: int,
) -> np.ndarray:
    if variant == "meanflow_original":
        return meanflow_actions.copy()
    if variant == "release_original":
        return release_actions.copy()
    if variant == "meanflow_motion_release_gripper":
        actions = meanflow_actions.copy()
        actions[:, -1] = _pad_or_trim(np.sign(release_actions[:, -1]), len(actions))
        return actions
    if variant == "release_motion_meanflow_gripper":
        actions = release_actions.copy()
        actions[:, -1] = _pad_or_trim(np.sign(meanflow_actions[:, -1]), len(actions))
        return actions
    if variant == "meanflow_motion_smoothed_gripper":
        actions = meanflow_actions.copy()
        actions[:, -1] = _smooth_gripper(actions[:, -1], min_run=smooth_min_run)
        return actions
    raise ValueError(f"Unknown variant: {variant}")


def _resolve_num_steps_wait(
    *,
    override: int | None,
    meanflow_trace: dict[str, Any],
    release_trace: dict[str, Any],
) -> int:
    if override is not None:
        return int(override)
    candidates = []
    for trace in (meanflow_trace, release_trace):
        config = trace.get("config", {})
        if "num_steps_wait" in config:
            candidates.append(int(config["num_steps_wait"]))
    if not candidates:
        return 5
    if len(set(candidates)) != 1:
        raise ValueError(f"Trace num_steps_wait mismatch: {candidates}")
    return candidates[0]


def _summarize_actions(actions: np.ndarray) -> dict[str, Any]:
    gripper = actions[:, -1]
    gripper_sign = np.sign(gripper)
    close_mask = gripper_sign > 0
    segments = _segments(gripper_sign)
    return {
        "n_actions": int(actions.shape[0]),
        "first_close_step": int(np.argmax(close_mask)) if bool(np.any(close_mask)) else None,
        "close_ratio": float(np.mean(close_mask)) if len(close_mask) else None,
        "gripper_transitions": max(0, len(segments) - 1),
        "gripper_segments": [[int(start), int(end), float(value)] for start, end, value in segments],
        "max_pos": float(np.max(np.abs(actions[:, :3]))) if len(actions) else None,
        "max_rot": float(np.max(np.abs(actions[:, 3:6]))) if len(actions) else None,
    }


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _write_trace(
    output_dir: Path,
    *,
    suite: str,
    task_id: int,
    trial: int,
    variant: str,
    success: bool,
    task_description: str,
    actions: np.ndarray,
    source_paths: dict[str, str],
) -> Path:
    trace_dir = output_dir / suite / "action_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "task_description": task_description,
        "suite": suite,
        "task_id": int(task_id),
        "trial": int(trial),
        "variant": variant,
        "success": bool(success),
        "source_paths": source_paths,
        "summary": _summarize_actions(actions),
        "executed_policy_actions": [
            {
                "action_idx": int(idx),
                "action": action.tolist(),
            }
            for idx, action in enumerate(actions)
        ],
    }
    path = trace_dir / f"variant={variant}--task{task_id}_trial{trial}_success{bool(success)}_action_trace.json"
    path.write_text(json.dumps(trace, indent=2, cls=NumpyEncoder), encoding="utf-8")
    return path


def _replay_actions(
    *,
    suite: str,
    task_id: int,
    trial: int,
    env,
    initial_state,
    task_description: str,
    variant: str,
    actions: np.ndarray,
    output_dir: Path,
    num_steps_wait: int,
    save_video: bool,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    max_steps = _get_max_steps(suite)
    replay_images = []
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            break
    done = False
    executed = []
    for action_idx, action in enumerate(actions[:max_steps]):
        if save_video:
            replay_images.append(get_libero_image(obs))
        obs, _, done, _ = env.step(action.tolist())
        executed.append(action)
        if done:
            break
    executed_actions = np.asarray(executed, dtype=np.float32)

    if executed_actions.size == 0:
        executed_actions = np.empty((0, actions.shape[1]), dtype=np.float32)

    if save_video:
        video_dir = output_dir / suite / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        save_rollout_video(
            video_dir,
            replay_images,
            f"variant={variant}--task{task_id}_trial{trial}",
            success=bool(done),
            task_description=task_description,
        )

    trace_path = _write_trace(
        output_dir,
        suite=suite,
        task_id=task_id,
        trial=trial,
        variant=variant,
        success=bool(done),
        task_description=task_description,
        actions=executed_actions,
        source_paths=source_paths,
    )

    row = {
        "suite": suite,
        "task_id": int(task_id),
        "trial": int(trial),
        "variant": variant,
        "success": bool(done),
        "n_executed": int(executed_actions.shape[0]),
        "trace_path": str(trace_path),
        "task_description": task_description,
    }
    row.update(_summarize_actions(executed_actions))
    return row


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, cls=NumpyEncoder), encoding="utf-8")

    fieldnames = [
        "suite",
        "task_id",
        "trial",
        "variant",
        "success",
        "n_executed",
        "first_close_step",
        "close_ratio",
        "gripper_transitions",
        "max_pos",
        "max_rot",
        "task_description",
        "trace_path",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    by_variant = {}
    for row in rows:
        stats = by_variant.setdefault(row["variant"], {"successes": 0, "trials": 0})
        stats["trials"] += 1
        stats["successes"] += int(bool(row["success"]))
    (output_dir / "variant_success.json").write_text(
        json.dumps(by_variant, indent=2, cls=NumpyEncoder),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--meanflow-run", type=Path, required=True)
    parser.add_argument("--release-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=None,
        help="Override dummy wait steps. By default this is read from trace config.",
    )
    parser.add_argument("--smooth-min-run", type=int, default=10)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tasks = _load_task_file(args.task_file)
    benchmark_dict = benchmark.get_benchmark_dict()

    rows = []
    for suite, task_id in tasks:
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if args.trial >= len(initial_states):
            raise ValueError(f"Trial {args.trial} is out of range for {suite} task{task_id}.")

        meanflow_trace, meanflow_actions = _load_trace_actions(args.meanflow_run, suite, task_id, args.trial)
        release_trace, release_actions = _load_trace_actions(args.release_run, suite, task_id, args.trial)
        env, env_task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=None)
        task_description = str(
            release_trace.get("task_description")
            or meanflow_trace.get("task_description")
            or env_task_description
            or ""
        )
        source_paths = {
            "meanflow": str(meanflow_trace["_trace_path"]),
            "release": str(release_trace["_trace_path"]),
        }
        num_steps_wait = _resolve_num_steps_wait(
            override=args.num_steps_wait,
            meanflow_trace=meanflow_trace,
            release_trace=release_trace,
        )

        try:
            for variant in args.variants:
                actions = _build_variant_actions(
                    variant,
                    meanflow_actions,
                    release_actions,
                    smooth_min_run=int(args.smooth_min_run),
                )
                row = _replay_actions(
                    suite=suite,
                    task_id=task_id,
                    trial=args.trial,
                    env=env,
                    initial_state=copy.deepcopy(initial_states[args.trial]),
                    task_description=task_description,
                    variant=variant,
                    actions=actions,
                    output_dir=args.output_dir,
                    num_steps_wait=num_steps_wait,
                    save_video=bool(args.save_video),
                    source_paths=source_paths,
                )
                rows.append(row)
                print(
                    f"{suite} task{task_id} {variant}: "
                    f"success={row['success']} executed={row['n_executed']}",
                    flush=True,
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    _write_summary(args.output_dir, rows)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
