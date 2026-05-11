import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

DEFAULT_TASKS = [
    "click_bell",
    "click_alarmclock",
    "adjust_bottle",
    "grab_roller",
    "beat_block_hammer",
    "dump_bin_bigbin",
    "blocks_ranking_size",
    "stack_blocks_two",
]

PHASE_TO_TASK_CONFIG = {
    "clean": "demo_clean",
    "random": "demo_randomized",
}


def _split_words(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split() if item.strip()]


def _resolve_path(path_str: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_str)))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _parse_success_rate(result_file: Path) -> float:
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return float(last_value)


def _mean(values: list[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class GroupConfig:
    tag: str
    ckpt: Path
    stats: Path
    steps: int

    @property
    def ckpt_tag(self) -> str:
        return _resolve_ckpt_tag(self.ckpt)


@dataclass(frozen=True)
class EvalJob:
    group: GroupConfig
    task_name: str
    phase: str
    run_ts: str

    @property
    def job_key(self) -> str:
        return f"{self.group.tag}__{self.phase}__{self.task_name}"


@dataclass
class RunningJob:
    job: EvalJob
    gpu_id: int
    process: subprocess.Popen[str]
    log_file: Path


def _build_groups() -> dict[str, GroupConfig]:
    release_ckpt = _resolve_path(
        os.environ.get(
            "RELEASE_CKPT",
            "./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt",
        )
    )
    stats = _resolve_path(
        os.environ.get(
            "STATS",
            "./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json",
        )
    )
    endpoint_ckpt = _resolve_path(
        os.environ.get(
            "ENDPOINT_CKPT",
            "./runs/robotwin_one_step_action_10step/checkpoints/weights/step_000010.pt",
        )
    )
    groups = {
        "release_1": GroupConfig("release_1", release_ckpt, stats, 1),
        "release_4": GroupConfig("release_4", release_ckpt, stats, 4),
        "release_10": GroupConfig("release_10", release_ckpt, stats, 10),
        "endpoint_1": GroupConfig("endpoint_1", endpoint_ckpt, stats, 1),
    }
    return groups


def main() -> None:
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    run_id = os.environ.get("RUN_ID", "quick_clean20")
    output_root = _resolve_path(
        os.environ.get(
            "OUTPUT_ROOT",
            f"./evaluate_results/robotwin_quick_one_step_compare/{run_id}",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = _split_words(os.environ.get("TASKS"), DEFAULT_TASKS)
    phases = _split_words(os.environ.get("PHASES"), ["clean"])
    for phase in phases:
        if phase not in PHASE_TO_TASK_CONFIG:
            raise ValueError(f"Unsupported phase: {phase}. Supported: {sorted(PHASE_TO_TASK_CONFIG)}")

    all_groups = _build_groups()
    group_names = _split_words(
        os.environ.get("GROUPS"),
        ["release_1", "release_4", "release_10", "endpoint_1"],
    )
    groups: list[GroupConfig] = []
    for group_name in group_names:
        if group_name not in all_groups:
            raise ValueError(f"Unknown group: {group_name}. Available: {sorted(all_groups)}")
        group = all_groups[group_name]
        if not group.ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for group {group.tag}: {group.ckpt}")
        if not group.stats.exists():
            raise FileNotFoundError(f"Stats not found for group {group.tag}: {group.stats}")
        groups.append(group)

    episodes = int(os.environ.get("EPISODES", "20"))
    num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    max_tasks_per_gpu = int(os.environ.get("MAX_TASKS_PER_GPU", "1"))
    task_config_name = os.environ.get("TASK", "robotwin_uncond_3cam_384_1e-4")
    if episodes <= 0:
        raise ValueError("EPISODES must be > 0")
    if num_gpus <= 0:
        raise ValueError("NUM_GPUS must be > 0")
    if max_tasks_per_gpu <= 0:
        raise ValueError("MAX_TASKS_PER_GPU must be > 0")

    jobs = deque(
        EvalJob(
            group=group,
            task_name=task_name,
            phase=phase,
            run_ts=f"{run_id}_{group.tag}_{phase}",
        )
        for group in groups
        for phase in phases
        for task_name in tasks
    )
    total_jobs = len(jobs)
    running: list[RunningJob] = []
    records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    manager_log = output_root / "manager.log"
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    failed_file = output_root / "failed_jobs.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def gpu_running_count(gpu_id: int) -> int:
        return sum(1 for item in running if item.gpu_id == gpu_id and item.process.poll() is None)

    def build_cmd(job: EvalJob, gpu_id: int) -> list[str]:
        return [
            sys.executable,
            str(SINGLE_ENTRY),
            f"task={task_config_name}",
            f"ckpt={str(job.group.ckpt)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={job.task_name}",
            f"EVALUATION.task_config={PHASE_TO_TASK_CONFIG[job.phase]}",
            f"EVALUATION.eval_num_episodes={episodes}",
            f"EVALUATION.num_inference_steps={job.group.steps}",
            f"EVALUATION.dataset_stats_path={str(job.group.stats)}",
            f"EVALUATION.output_dir={str(output_root / job.run_ts)}",
        ]

    def launch(job: EvalJob, gpu_id: int) -> RunningJob:
        cmd = build_cmd(job, gpu_id)
        log_file = logs_dir / f"{job.job_key}.log"
        log(f"launch {job.job_key} gpu={gpu_id} cmd={' '.join(cmd)}")
        f = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Keep the file handle alive through the child process by storing it on the process object.
        process._fastwam_log_handle = f  # type: ignore[attr-defined]
        return RunningJob(job=job, gpu_id=gpu_id, process=process, log_file=log_file)

    def close_log_handle(process: subprocess.Popen[str]) -> None:
        handle = getattr(process, "_fastwam_log_handle", None)
        if handle is not None:
            handle.close()

    def terminate_all() -> None:
        for item in running:
            if item.process.poll() is None:
                log(f"terminate {item.job.job_key} gpu={item.gpu_id}")
                item.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for item in running:
            if item.process.poll() is not None:
                close_log_handle(item.process)
                continue
            try:
                item.process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                log(f"kill {item.job.job_key} gpu={item.gpu_id}")
                item.process.kill()
                item.process.wait()
            close_log_handle(item.process)

    def try_launch_pending() -> None:
        for gpu_id in range(num_gpus):
            while jobs and gpu_running_count(gpu_id) < max_tasks_per_gpu:
                running.append(launch(jobs.popleft(), gpu_id))

    def write_outputs() -> None:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group",
                    "checkpoint",
                    "ckpt_tag",
                    "num_inference_steps",
                    "phase",
                    "task_name",
                    "episodes",
                    "success_rate",
                    "result_file",
                    "log_file",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            key = f"{record['group']}::{record['phase']}"
            grouped.setdefault(
                key,
                {
                    "group": record["group"],
                    "phase": record["phase"],
                    "num_inference_steps": record["num_inference_steps"],
                    "tasks": [],
                },
            )
            grouped[key]["tasks"].append(record)
        payload = {
            "run_id": run_id,
            "episodes": episodes,
            "tasks": tasks,
            "phases": phases,
            "groups": [
                {
                    **value,
                    "mean_success_rate": _mean([float(r["success_rate"]) for r in value["tasks"]]),
                }
                for value in grouped.values()
            ],
            "failed": failed_records,
        }
        summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with failed_file.open("w", encoding="utf-8") as f:
            for record in failed_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(
        f"quick compare start run_id={run_id} groups={group_names} tasks={len(tasks)} "
        f"phases={phases} episodes={episodes} num_gpus={num_gpus} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} total_jobs={total_jobs}"
    )

    has_failure = False
    failure_message = ""
    try_launch_pending()
    while running:
        progressed = False
        for item in list(running):
            return_code = item.process.poll()
            if return_code is None:
                continue
            progressed = True
            running.remove(item)
            close_log_handle(item.process)

            job = item.job
            if return_code != 0:
                has_failure = True
                failure_message = f"worker failed: {job.job_key} gpu={item.gpu_id} return_code={return_code}"
                failed_records.append(
                    {
                        "job": job.job_key,
                        "gpu_id": item.gpu_id,
                        "return_code": return_code,
                        "log_file": str(item.log_file),
                    }
                )
                log(failure_message)
                terminate_all()
                break

            result_file = (
                PROJECT_ROOT
                / "evaluate_results"
                / "robotwin"
                / job.group.ckpt_tag
                / job.run_ts
                / job.task_name
                / _phase_result_filename(job.phase)
            )
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = f"result parse failed: {job.job_key} error={repr(exc)}"
                failed_records.append(
                    {
                        "job": job.job_key,
                        "gpu_id": item.gpu_id,
                        "return_code": return_code,
                        "reason": repr(exc),
                        "log_file": str(item.log_file),
                        "result_file": str(result_file),
                    }
                )
                log(failure_message)
                terminate_all()
                break

            record = {
                "group": job.group.tag,
                "checkpoint": str(job.group.ckpt),
                "ckpt_tag": job.group.ckpt_tag,
                "num_inference_steps": job.group.steps,
                "phase": job.phase,
                "task_name": job.task_name,
                "episodes": episodes,
                "success_rate": success_rate,
                "result_file": str(result_file),
                "log_file": str(item.log_file),
            }
            records.append(record)
            log(f"done {job.job_key} gpu={item.gpu_id} success_rate={success_rate:.4f}")
            write_outputs()
            try_launch_pending()

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    if has_failure:
        write_outputs()
        raise RuntimeError(failure_message)

    write_outputs()
    log(f"quick compare finished summary={summary_csv} json={summary_json}")


if __name__ == "__main__":
    main()
