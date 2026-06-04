import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env, get_libero_image
from libero.libero import benchmark


def _obs_summary(obs: dict) -> dict:
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    return {key: np.asarray(obs[key], dtype=np.float32).round(6).tolist() for key in keys if key in obs}


def _step(env, action, label: str, capture_images: bool):
    arr = np.asarray(action, dtype=np.float32)
    print(
        f"[replay] before {label} action={arr.round(6).tolist()} "
        f"max_abs={float(np.max(np.abs(arr))):.6f}",
        flush=True,
    )
    obs, _, done, _ = env.step(arr)
    if capture_images:
        imgs = get_libero_image(obs)
        image_shapes = {key: list(value.shape) for key, value in imgs.items()}
        # Copy to mimic eval_libero_single.py replay frame retention.
        _ = {key: value.copy() for key, value in imgs.items()}
        print(f"[replay] captured images {label} shapes={image_shapes}", flush=True)
    print(f"[replay] after {label} done={bool(done)} obs={_obs_summary(obs)}", flush=True)
    return obs, bool(done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a LIBERO action trace without loading a policy model.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--trial-idx", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-before-env-t", type=int, default=None)
    parser.add_argument("--capture-images", action="store_true")
    args = parser.parse_args()

    with open(args.trace, "r", encoding="utf-8") as f:
        trace = json.load(f)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    print(
        f"[replay] task={args.task_suite}/{args.task_id} trial={args.trial_idx} "
        f"description={task_description!r}",
        flush=True,
    )
    env.reset()
    obs = env.set_init_state(initial_states[args.trial_idx])
    print(f"[replay] initial obs={_obs_summary(obs)}", flush=True)

    dummy_actions = sorted(trace.get("dummy_wait_actions", []), key=lambda record: int(record["env_t"]))
    policy_actions = sorted(trace.get("executed_policy_actions", []), key=lambda record: int(record["env_t"]))

    for record in dummy_actions:
        env_t = int(record["env_t"])
        if args.stop_before_env_t is not None and env_t >= args.stop_before_env_t:
            print(f"[replay] stop before env_t={env_t}", flush=True)
            return
        obs, done = _step(
            env,
            record["action"],
            label=f"dummy env_t={env_t}",
            capture_images=args.capture_images,
        )
        if done:
            print(f"[replay] done during dummy env_t={env_t}", flush=True)
            return

    for record in policy_actions:
        env_t = int(record["env_t"])
        if args.stop_before_env_t is not None and env_t >= args.stop_before_env_t:
            print(f"[replay] stop before env_t={env_t}", flush=True)
            return
        label = (
            f"policy env_t={env_t} replan={record.get('replan_idx')} "
            f"idx={record.get('action_idx_in_replan')}"
        )
        obs, done = _step(env, record["action"], label=label, capture_images=args.capture_images)
        if done:
            print(f"[replay] done during policy env_t={env_t}", flush=True)
            return

    print("[replay] completed all trace actions without done or abort", flush=True)


if __name__ == "__main__":
    main()
