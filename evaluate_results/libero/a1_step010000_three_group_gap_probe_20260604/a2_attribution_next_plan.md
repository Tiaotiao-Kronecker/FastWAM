# A2 attribution next plan

Date recorded: 2026-06-05

## Source plan

- `docs/meanflow_action_a2_design.html`, section 11, A1/A2 attribution training plan.
- `docs/plan/2026-06-03-meanflow-action-a2-design-plan.md`, Phase 1: A1-continue attribution.

## Current completed Phase-1 branches

All completed branches start from the same A1 70k checkpoint and keep A1 rank,
target modules, data distribution, LR policy, eval style, and endpoint disabled.

| branch | status | gap-probe result | readout |
|---|---:|---:|---|
| `A1-continue-control` | complete | 320/350 | Continuing A1 for 10k does not improve A1 70k; it drops 13 successes on the seven-task gap probe. |
| `A1.2-residual-only` | complete | 320/350 | Residual target is not a catastrophic code-path failure, but it does not improve long-horizon tasks. |
| `A1.3-clip0.25` | complete | 317/350 | Standalone clipping at token-L2 max norm 0.25 does not pass the attribution gate. |

Reference on the same seven-task gap probe: A1 step070k is 333/350.

## Interpretation against the A2 plan

The three completed 10k pilots do not meet the promotion rule from the A2 plan:
hard LIBERO-10 tasks should improve while A1 strong/control tasks should not
meaningfully regress. The current gap-probe does not include LIBERO-10 tasks
4/6/9, so it is a partial readout rather than the final A2 attribution probe.

Do not promote `A1.2-residual-only` or `A1.3-clip0.25` to 30k. Do not start
endpoint ablation yet, because `A1.5-endpoint` is defined as an add-on to the
winning minimal objective, and no objective has won yet.

## Immediate execution plan

1. Add the plan-aligned probe task set:
   `experiments/libero/task_sets/libero_a2_attribution_probe_v1.txt`.
2. Re-evaluate existing checkpoints on that probe:
   - A1 step070k baseline
   - `A1-continue-control` step010k
   - `A1.2-residual-only` step010k
   - `A1.3-clip0.25` step010k
3. Implement/smoke `A1.4-mix` once the interval mixture sampler/config is
   available.
4. Run 10k `A1.4-mix` and `A1.4-mix+clip` from A1 70k, keeping rank 4,
   A1 target modules, data distribution, LR policy, and endpoint disabled.
5. Promote only the branch that improves the hard tasks and preserves controls
   to 30k. Run endpoint/data/capacity/release-clean only after a minimal
   objective wins Phase 1.

## Evaluation caveats

For the next readout, keep `EVALUATION.policy_subprocess` consistent across all
groups. Earlier final serial runs for control and residual-only used
`policy_subprocess=false`, while residual-clip used `true`.

## Execution artifacts added on 2026-06-05

- `experiments/libero/task_sets/libero_a2_attribution_probe_v1.txt`
- `evaluate_results/libero/a1_step010000_three_group_gap_probe_20260604/run_a2_attribution_probe_existing_ckpts.sh`
- `configs/task/libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5.yaml`
- `configs/task/libero_one_step_meanflow_a1_mix_clip_lora_eqanchor_2cam224_5e-5.yaml`
- `evaluate_results/libero/a1_step010000_three_group_gap_probe_20260604/run_a1_4_mix_attribution_training.sh`

Current GPU state on H200-2 is saturated by other jobs, so eval/training launch
should wait for at least one free GPU unless explicitly preempting external
workloads.
