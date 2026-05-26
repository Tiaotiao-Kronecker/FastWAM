# 2026-05-22 LIBERO MeanFlow Conditioner Ablation Worklog

## Purpose

Diagnose why previous FastWAM MeanFlow attempts failed on LIBERO despite
finite-difference training and apparently finite losses.

The key user hypothesis was that the existing MeanFlow start conditioning was
too weak:

```text
final action t_mod = original time_mod(t) + meanflow_start_mod(r)
```

This looked suspicious because `r` entered only as a simple additive residual
to the original action DiT time modulation. Previous action-only runs were also
not clean diagnostics because they trained the full action expert and proprio
encoder.

## Clean Diagnostic Setup

The clean setup intentionally trains only a small MeanFlow conditioner while
preserving the release policy as much as possible:

```yaml
objective: finite_difference
random_timesteps: true
equal_time_prob: 0.0
trainable_scope: conditioner
train_proprio_encoder: false
freeze_video_expert: true
derivative_epsilon: 0.05
loss.lambda_meanflow_target: 1.0
loss.lambda_action_velocity: 0.0
loss.lambda_action_endpoint: 0.0
```

Frozen components:

```text
video expert
action expert body, including DiT attention/MLP/head parameters
proprio encoder
```

Trainable components in the winning setting:

```text
action_expert.meanflow_joint_time_embedding
action_expert.meanflow_joint_time_projection
```

## Conditioner Modes

### A: additive_start

This is the original start-conditioning style:

```text
start_emb = emb(r)
start_mod = MLP(start_emb) -> [B, 6, hidden_dim]
final action t_mod = original time_mod(t) + start_mod
```

It only sees the start timestep `r`. At inference `r=0`, so this can behave
close to a fixed residual offset.

### B: joint_delta

This is the new diagnostic/winning mode:

```text
action_emb   = emb(t)
start_emb    = emb(r)
interval_emb = emb(t - r)
joint_emb    = concat(action_emb, start_emb, interval_emb)
joint_mod    = MLP(joint_emb) -> [B, 6, hidden_dim]
final action t_mod = original time_mod(t) + joint_mod
```

With the current action expert config:

```text
freq_dim = 256
hidden_dim = 1024
joint_emb dim = 256 * 3 = 768
joint_mod shape = [B, 6, 1024]
```

The final projection is zero-initialized, so training starts from release-like
behavior:

```text
joint_mod = 0
final action t_mod = original time_mod(t)
```

The important distinction is that `joint_delta` still uses residual addition,
but the residual is a function of the full MeanFlow interval:

```text
joint_mod = f(t, r, t-r)
```

not just `f(r)`.

## MeanFlow Variables

MeanFlow learns:

```text
u_theta(x_t, t, r | image, text, proprio)
```

where:

```text
t = current noisy action timestep
r = target/start timestep
t-r = interval length
```

Training samples random intervals:

```text
0 <= r < t <= 1000
```

Inference uses one-step action generation with:

```text
r = 0
NUM_INFERENCE_STEPS = 1
```

This means each policy call uses one denoising step. A full LIBERO episode still
has repeated policy calls/replans across the rollout.

## A/B Training Runs

A additive:

```text
runs/libero_one_step_meanflow_d0_conditioner_2cam224_1e-4/additive_cond_fd_2k_20260522
```

B joint:

```text
runs/libero_one_step_meanflow_joint_conditioner_2cam224_1e-4/joint_cond_fd_2k_20260522
```

Both were trained from:

```text
./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

Both trained for 2000 steps and saved checkpoints every 500 steps.

Final losses were finite:

```text
additive step2000 loss = 0.0122
joint step2000 loss    = 0.0672
```

No numerical explosion was observed.

## Gap Probe Evaluation

Evaluation was aligned to previous release/MeanFlow diagnostics:

```text
TASK_SET=libero_gap_probe_v1
TASK_FILE=experiments/libero/task_sets/libero_gap_probe_v1.txt
TRIAL_INDICES=[0]
NUM_TRIALS=1
NUM_INFERENCE_STEPS=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
```

Results:

| Checkpoint | additive_start | additive failures | joint_delta | joint failures |
| --- | ---: | --- | ---: | --- |
| step 500 | 6/7 | libero_10_2 | 6/7 | libero_10_0 |
| step 1000 | 5/7 | libero_10_0, libero_10_2 | 7/7 | none |
| step 1500 | 5/7 | libero_10_0, libero_10_2 | 7/7 | none |
| step 2000 | 5/7 | libero_10_0, libero_10_2 | 7/7 | none |

Reference context:

```text
release gap probe: 6/7 or 7/7 depending release inference steps
old action-only finite-difference MeanFlow: 1/7
joint finite-difference MeanFlow: 0/7
C-joint finite-difference checkpoints 500/1000/1500/2000: 0/7
```

## Action Trace Observation

Trace summary for additive step2000 vs joint step2000 was written to:

```text
evaluate_results/libero/mf_conditioner_ab_trace_summary_20260522/action_trace_summary_executed_policy_actions.csv
```

Key failure contrast:

```text
libero_10_0:
  additive failed, 700 executed actions, 34 gripper transitions
  joint succeeded, 337 executed actions, 8 gripper transitions

libero_10_2:
  additive failed, 700 executed actions, 20 gripper transitions
  joint succeeded, 229 executed actions, 4 gripper transitions
```

The additive failures look like unstable timing/control behavior rather than
numeric blow-up.

## Interpretation

The old action-only failure cannot be attributed only to simple residual
addition, because the clean additive conditioner recovered to 6/7 at step500.

The more likely issue with old action-only was excessive trainable scope:

```text
full action expert trained
proprio encoder trained
release policy was not preserved
```

However, the simple `f(r)` additive conditioner is not stable enough. It drops
to 5/7 and repeatedly fails the same LIBERO-10 tasks after step1000.

The joint conditioner is the current best candidate because:

```text
it preserves release weights
it trains only a small MeanFlow interval conditioner
it conditions on t, r, and t-r
it reaches 7/7 from step1000 through step2000
```

Preferred checkpoint after the single-trial gap probe only:

```text
joint_delta step1000
```

Reason: step1000 is the earliest checkpoint that reaches 7/7 on the aligned
gap probe, and later checkpoints do not improve that diagnostic. This was a
preliminary conclusion and is superseded by the multi-trial and official-long
results below.

## Next Validation Plan

1. Run multi-trial gap probe for joint_delta step1000/1500/2000:

```text
TRIAL_INDICES=[0,1,2,3,4]
NUM_TRIALS=5
NUM_INFERENCE_STEPS=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
```

2. If step1000 remains stable, use it as the main candidate.

3. Run a larger release-aligned task set with the selected checkpoint.

4. Keep additive_start only as an ablation baseline, not as the main line.

## Multi-Trial Gap Probe Follow-Up

The first follow-up used the same biased gap probe, but with five trials:

```text
TASK_SET=libero_gap_probe_v1
TRIAL_INDICES=[0,1,2,3,4]
NUM_TRIALS=5
NUM_INFERENCE_STEPS=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
```

Results:

| Checkpoint | Overall | Spatial | Goal | LIBERO-10 | Main failures |
| --- | ---: | ---: | ---: | ---: | --- |
| step1000 | 31/35 = 88.57% | 100% | 90% | 85% | libero_10_0, libero_10_8, goal_6 |
| step1500 | 33/35 = 94.29% | 100% | 90% | 95% | libero_10_8, goal_5 |
| step2000 | 33/35 = 94.29% | 100% | 90% | 95% | libero_10_3, goal_5 |

This changes the checkpoint selection:

```text
single-trial gap probe suggested step1000 was enough
multi-trial gap probe shows step1000 is not the most stable checkpoint
step1500 and step2000 tie on 5-trial gap probe
```

Because step1500 and step2000 tie, the next validation should compare both on
the cleaner release-aligned long-horizon task set:

```text
TASK_SET=libero_long_official
TRIAL_INDICES=[0,1,2,3,4]
NUM_TRIALS=5
NUM_INFERENCE_STEPS=1
```

## Official Long Follow-Up

The second follow-up used all ten official `libero_10` tasks, still with one
inference step and the same five trial indices:

```text
TASK_SET=libero_long_official
TRIAL_INDICES=[0,1,2,3,4]
NUM_TRIALS=5
NUM_INFERENCE_STEPS=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
```

Outputs:

```text
evaluate_results/libero/mf_jointcond_step1500_long_official_t0-4_20260522
evaluate_results/libero/mf_jointcond_step2000_long_official_t0-4_20260522
```

Results:

| Checkpoint | Overall | Main failures |
| --- | ---: | --- |
| step1500 | 47/50 = 94.00% | libero_10_1, libero_10_6, libero_10_8 each 80% |
| step2000 | 48/50 = 96.00% | libero_10_1 and libero_10_3 each 80% |

No runner failures were recorded in either `failed_tasks.txt`.

This changes the checkpoint selection again:

```text
step1000 is too weak on multi-trial gap probe
step1500 and step2000 tie on multi-trial gap probe
step2000 is better on official-long: 96.00% vs 94.00%
```

Current preferred checkpoint:

```text
joint_delta step2000
```

The next validation should use `step2000` on `libero_long_horizon_v1`, which
adds the high-gap standard tasks `libero_goal,5`, `libero_goal,6`, and
`libero_spatial,3` to the official `libero_10` set. This is not a paper-style
benchmark slice, but it is a stronger diagnostic for one-step degradation than
the saturated official-long set.

## Long Horizon Diagnostic Follow-Up

The third follow-up used the broader diagnostic set:

```text
TASK_SET=libero_long_horizon_v1
TRIAL_INDICES=[0,1,2,3,4]
NUM_TRIALS=5
NUM_INFERENCE_STEPS=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
```

Output:

```text
evaluate_results/libero/mf_jointcond_step2000_long_horizon_v1_t0-4_20260522
```

Result:

| Suite | Success |
| --- | ---: |
| libero_spatial | 5/5 = 100.00% |
| libero_goal | 9/10 = 90.00% |
| libero_10 | 48/50 = 96.00% |
| overall | 62/65 = 95.38% |

Task-level failures:

```text
libero_10_1 trial1 failed
libero_10_3 trial1 failed
libero_goal_5 trial3 failed
```

Action trace diagnostics were written to:

```text
evaluate_results/libero/mf_jointcond_step2000_long_horizon_v1_trace_summary_20260522
```

The failed `libero_10` episodes ran to 700 executed actions, while the failed
`libero_goal_5` episode ran to 400 executed actions. This looks like ordinary
rollout failure on a small number of trials, not a runner failure or numerical
collapse.

Final recommendation from this diagnostic sequence:

```text
Use joint_delta step2000 as the current MeanFlow candidate.
Keep additive_start as an ablation baseline only.
Do not return to action-only/full-action-expert training unless there is a
separate reason to test policy-body finetuning, because the clean conditioner
diagnostic indicates preserving the release policy is the key stability factor.
```
