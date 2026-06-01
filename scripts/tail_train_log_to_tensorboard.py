import argparse
import os
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step=(\d+)/(\d+)")
METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
SAMPLES_RE = re.compile(r",\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+samples/s")


TAG_MAP = {
    "loss": "train/loss",
    "lr": "train/lr",
    "speed": "performance/steps_per_sec",
    "samples_per_sec": "performance/samples_per_sec",
}


def clean(line: str) -> str:
    return ANSI_RE.sub("", line)


def tag_for(key: str) -> str:
    if key in TAG_MAP:
        return TAG_MAP[key]
    if key.startswith("loss_") or key.startswith("meanflow_") or key.startswith("pred_") or key.startswith("target_"):
        return f"train/{key}"
    if key.startswith("equal_time_"):
        return f"train/{key}"
    return f"train/{key}"


def parse_block(lines: list[str]) -> tuple[int | None, int | None, dict[str, float]]:
    text = "\n".join(clean(line) for line in lines)
    step_match = STEP_RE.search(text)
    if step_match is None:
        return None, None, {}
    step = int(step_match.group(1))
    total = int(step_match.group(2))
    metrics: dict[str, float] = {}
    for key, value in METRIC_RE.findall(text):
        if key in {"epoch", "step", "eta"}:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    samples_match = SAMPLES_RE.search(text)
    if samples_match is not None:
        metrics["samples_per_sec"] = float(samples_match.group(1))
    if total > 0:
        metrics["progress_fraction"] = step / total
    return step, total, metrics


def iter_blocks_from_text(text: str):
    block: list[str] = []
    for line in text.splitlines():
        if STEP_RE.search(line):
            if block:
                yield block
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield block


def write_blocks(writer: SummaryWriter, text: str, seen_steps: set[int]) -> int:
    written = 0
    for block in iter_blocks_from_text(text):
        step, _, metrics = parse_block(block)
        if step is None or step in seen_steps:
            continue
        for key, value in metrics.items():
            writer.add_scalar(tag_for(key), value, step)
        writer.flush()
        seen_steps.add(step)
        written += 1
        print(f"[tb] step={step} metrics={len(metrics)}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail a FastWAM rich log and mirror scalar metrics to TensorBoard.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--tb-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    tb_dir = Path(args.tb_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    seen_steps: set[int] = set()
    offset = 0

    print(f"[tb] reading {log_file}", flush=True)
    print(f"[tb] writing {tb_dir}", flush=True)

    while True:
        if not log_file.exists():
            time.sleep(args.poll_seconds)
            continue
        size = log_file.stat().st_size
        if size < offset:
            offset = 0
            seen_steps.clear()
        if size > offset:
            with log_file.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            write_blocks(writer, chunk, seen_steps)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
