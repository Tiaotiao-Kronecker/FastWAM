#!/usr/bin/env python
"""Collect prompts touched by the deterministic Robotwin training sampler."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _train_episode_order(total_episodes: int, val_set_proportion: float, seed: int) -> list[int]:
    episode_indices = list(range(total_episodes))
    if val_set_proportion < 1e-6:
        return episode_indices

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_indices)
    split_idx = int(total_episodes * (1 - val_set_proportion))
    return episode_indices[:split_idx]


def _sample_indices(num_frames: int, seed: int, count: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(num_frames, generator=generator)[:count].tolist()


def _episode_to_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    rel_path = str(info["data_path"]).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return dataset_dir / rel_path


def collect_prompts(
    dataset_dir: Path,
    output_path: Path,
    sample_count: int,
    seed: int,
    val_set_proportion: float,
) -> dict:
    info = _read_json(dataset_dir / "meta" / "info.json")
    episodes_rows = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    tasks_rows = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")

    episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes_rows}
    tasks = {int(row["task_index"]): str(row["task"]) for row in tasks_rows}

    train_episodes = _train_episode_order(
        total_episodes=int(info["total_episodes"]),
        val_set_proportion=val_set_proportion,
        seed=seed,
    )
    train_lengths = [episode_lengths[episode_index] for episode_index in train_episodes]
    cumulative = np.cumsum(train_lengths, dtype=np.int64)
    train_num_frames = int(cumulative[-1])

    sampled_indices = _sample_indices(train_num_frames, seed=seed, count=sample_count)
    episode_offsets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sampled_position, sample_idx in enumerate(sampled_indices):
        train_episode_position = bisect.bisect_right(cumulative, sample_idx)
        episode_start = 0 if train_episode_position == 0 else int(cumulative[train_episode_position - 1])
        episode_index = int(train_episodes[train_episode_position])
        local_frame_index = int(sample_idx - episode_start)
        episode_offsets[episode_index].append((sampled_position, local_frame_index))

    prompts_by_sample = [None] * len(sampled_indices)
    task_index_counts = Counter()
    for episode_index in sorted(episode_offsets):
        parquet_path = _episode_to_path(dataset_dir, info, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        task_index_column = pq.read_table(parquet_path, columns=["task_index"])["task_index"].to_numpy()
        for sampled_position, local_frame_index in episode_offsets[episode_index]:
            task_index = int(task_index_column[local_frame_index])
            task_index_counts[task_index] += 1
            task = tasks[task_index]
            prompts_by_sample[sampled_position] = DEFAULT_PROMPT.format(task=task)

    unique_prompts = OrderedDict()
    for prompt in prompts_by_sample:
        if prompt is None:
            raise RuntimeError("Internal error: prompt collection left an empty slot.")
        unique_prompts.setdefault(prompt, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prompt in unique_prompts:
            f.write(prompt)
            f.write("\n")

    stats = {
        "dataset_dir": str(dataset_dir),
        "output_path": str(output_path),
        "seed": seed,
        "val_set_proportion": val_set_proportion,
        "sample_count": sample_count,
        "train_num_episodes": len(train_episodes),
        "train_num_frames": train_num_frames,
        "unique_prompt_count": len(unique_prompts),
        "unique_episode_count": len(episode_offsets),
        "unique_task_index_count": len(task_index_counts),
        "first_sampled_indices": sampled_indices[:10],
        "most_common_task_indices": task_index_counts.most_common(10),
    }
    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    stats["stats_path"] = str(stats_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("./data/robotwin2.0/robotwin2.0"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-set-proportion", type=float, default=0.01)
    args = parser.parse_args()

    stats = collect_prompts(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        sample_count=args.sample_count,
        seed=args.seed,
        val_set_proportion=args.val_set_proportion,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
