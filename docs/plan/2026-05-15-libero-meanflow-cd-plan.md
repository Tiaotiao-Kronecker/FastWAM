# 2026-05-15 LIBERO MeanFlow C/D Plan

## Goal

Evaluate paper-style MeanFlow inside FastWAM while keeping the release comparison clean.

The main question is not whether MeanFlow can reduce training loss.
The main question is why MeanFlow rollout quality is worse than release:

```text
Does MeanFlow mainly fail because it weakens gripper/timing semantics,
because action-only fine-tuning breaks the joint video/action structure,
or because the 2k pilot is simply undertrained?
```

The anchor remains:

```text
checkpoint: checkpoints/fastwam_release/libero_uncond_2cam224.pt
stats:      checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

All LIBERO 2cam training now defaults to the release stats through
`configs/data/libero_2cam.yaml`, so fine-tuning stays in the same normalized
action/state coordinate system as `release_1` and `release_10`.

## Implemented Objectives

`src/fastwam/models/wan22/fastwam_one_step_meanflow.py` now supports:

```text
objective: paper_jvp
random_timesteps: true
equal_time_prob: 0.25

z_t = (1 - t) * action + t * noise
v   = noise - action
u, du_dt = JVP(u_theta, primals=(z_t, t, r), tangents=(v, 1, 0))
u_target = v - (t - r) * du_dt
loss = MSE(u, stopgrad(u_target))
```

The older finite-difference implementation remains available with:

```text
model.one_step_meanflow.objective=finite_difference
```

The JVP path locally disables activation checkpointing and uses a manual
attention fallback because PyTorch forward-mode AD cannot pass through
`torch.utils.checkpoint` or flash SDPA in this environment.

It also supports the current pilot objective:

```text
objective: finite_difference
random_timesteps: true
equal_time_prob: 0.0

sample r < t
z_t = (1 - t) * action + t * noise
v   = noise - action
u_t = u_theta(z_t, t, r)
z_prev = z_t - eps * v
u_prev = stopgrad(u_theta(z_prev, t - eps, r))
du_dt ~= (u_t - u_prev) / eps
u_target = v - (t - r) * du_dt
loss = MSE(u_t, stopgrad(u_target))
```

Finite difference is not the exact paper derivative, but it keeps the MeanFlow
target form while preserving the normal FastWAM bf16/checkpoint/flash-attention
training path.

Execution note on JVP:

```text
bf16 + torch.func.jvp currently fails before the first training step because
forward-mode dual tensors are promoted to fp32 while the large FastWAM Linear
weights are bf16. Running paper-JVP therefore requires mixed_precision=no.

This does not change the MeanFlow objective, release checkpoint, release stats,
trainable scope, or sampled r/t distribution. It changes only numeric precision
and resource cost. The measured FP32 JVP smoke took roughly 38 seconds for one
step, making a 2k run roughly 22-24 hours.

Decision: do not use JVP for the current C pilot. Keep JVP as a strict-paper
research path, and use finite_difference for the practical low-intrusion C/D
pilots.
```

## Experiment Rows

### C-action-only

Purpose: lowest-variable paper-style MeanFlow test against the release checkpoint.

```text
task=libero_one_step_meanflow_2cam224_1e-4
trainable_scope=action
train_proprio_encoder=true
freeze_video_expert=true
loss: meanflow target only
```

Interpretation:

```text
Does changing the action objective to MeanFlow improve one-step action generation
when release visual/world representations are fixed?
```

### D0-conditioner

Purpose: small-module ablation.

```text
task=libero_one_step_meanflow_d0_conditioner_2cam224_1e-4
trainable_scope=conditioner
train_proprio_encoder=false
freeze_video_expert=true
```

Interpretation:

```text
Can the release action backbone expose interval mean velocity using only new r/t conditioning?
```

### D1-conditioner-head

Purpose: slightly larger small-module ablation.

```text
task=libero_one_step_meanflow_d1_conditioner_head_2cam224_1e-4
trainable_scope=conditioner_head
train_proprio_encoder=false
freeze_video_expert=true
```

Interpretation:

```text
Does adding action head capacity materially improve over conditioner-only?
```

### C-joint, not implemented yet

This is the more FastWAM-faithful version:

```text
load release checkpoint
unfreeze video expert + action expert + proprio encoder + MeanFlow conditioning
video branch: original FastWAM video flow loss
action branch: paper-style MeanFlow loss
```

It requires a separate joint training loss instead of the current action-only
MeanFlow loss. It should run only if C-action-only gives a useful signal.

## Archived Execution Plan

Decision archived on 2026-05-15:

```text
Do not implement C-joint before seeing the C-action-only pilot signal.
Run the lowest-variable C pilot first, then use that result to decide whether
the extra joint video/action training code is worth adding.
```

Execution order:

1. Smoke each implemented row with `max_steps=1`.
2. Run C-action-only for 2k steps.
3. Check whether C-action-only produces a useful one-step action signal versus
   `release_1`.
4. Run D0/D1 2k pilots as diagnostic ablations.
5. Implement C-joint only if C-action-only shows signal, or if we explicitly
   need the second-stage test of FastWAM's coupled video/action training
   hypothesis.
6. Run final formal evaluation together for the surviving rows:

```text
release_1
release_4
release_10
C_meanflow_1
D0_meanflow_1, if pilot survives
D1_meanflow_1, if pilot survives
C_joint_meanflow_1, only if implemented after C signal
```

This keeps the comparison clean: C answers the low-variable objective-change
question first, while C-joint is reserved for the more expensive architectural
question.

## Multi-GPU

The trainer uses Accelerate and supports multi-process training. Existing launch
wrappers are:

```text
bash scripts/train_zero1.sh <num_gpus> task=...
bash scripts/train_zero2.sh <num_gpus> task=...
```

For JVP MeanFlow, prefer ZeRO-2 because the JVP path disables activation
checkpointing locally and uses manual attention inside JVP:

```bash
HF_HOME=/DATA/disk3/tmp/hf_home \
HF_DATASETS_CACHE=/DATA/disk3/tmp/hf_home/datasets \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_zero2.sh 4 \
  task=libero_one_step_meanflow_2cam224_1e-4 \
  max_steps=2000 \
  mixed_precision=no \
  save_training_state=false
```

For D0/D1, use the corresponding task names.

Current finite-difference C pilot:

```bash
HF_HOME=/DATA/disk3/tmp/hf_home \
HF_DATASETS_CACHE=/DATA/disk3/tmp/hf_home/datasets \
CUDA_VISIBLE_DEVICES=0,2,3,5 \
RUN_ID=meanflow_c_action_fd_bf16_2k_20260515 \
bash scripts/train_zero2.sh 4 \
  task=libero_one_step_meanflow_2cam224_1e-4 \
  max_steps=2000 \
  save_training_state=false
```

Observed at step 10:

```text
loss=0.9984
loss_meanflow_target=0.9984
meanflow_sigma_start=0.2867
meanflow_sigma_end=0.6118
meanflow_interval=0.3247
speed=0.23 step/s
eta=02:25:28
```

## Current Verification

Completed:

```text
python -m py_compile src/fastwam/models/wan22/fastwam_one_step_meanflow.py src/fastwam/models/wan22/wan_video_dit.py src/fastwam/runtime.py
Hydra --cfg job for C/D0/D1
C-action-only max_steps=1 smoke
C-action-only FP32 CUDA/ZeRO-2 max_steps=1 smoke
C-action-only finite-difference bf16 CUDA/ZeRO-2 max_steps=1 smoke
C-action-only finite-difference bf16 2k pilot started
```

The smoke checkpoint was saved at:

```text
runs/libero_one_step_meanflow_c_action_paper_jvp_smoke_20260515/checkpoints/weights/step_000001.pt
```

The FP32 CUDA/ZeRO-2 smoke checkpoint was saved at:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_paper_jvp_fp32_smoke_20260515/checkpoints/weights/step_000001.pt
```

Failed bf16 smoke attempts:

```text
/DATA/disk3/tmp/meanflow_c_action_paper_jvp_2k_20260515.log
/DATA/disk3/tmp/meanflow_c_action_paper_jvp_gpu_smoke_20260515.log
/DATA/disk3/tmp/meanflow_c_action_paper_jvp_gpu_smoke2_20260515.log
```

They failed before step 1 with:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

Active C pilot command uses:

```text
RUN_ID=meanflow_c_action_fd_bf16_2k_20260515
log=/DATA/disk3/tmp/meanflow_c_action_fd_bf16_2k_20260515.log
task=libero_one_step_meanflow_2cam224_1e-4
max_steps=2000
mixed_precision=bf16
```

## Post-2k Gate

Do not go directly from the 2k C pilot into full training. The 2k run is a
gate for the MeanFlow finite-difference C path, not a final model.

After the 2k checkpoint is available:

1. Run inference latency on the C checkpoint with the same synthetic-input
   benchmark used for release 1 vs release 10. The benchmark can load the
   MeanFlow task config and will exercise the MeanFlow
   `_predict_action_noise_with_cache` override.
2. Run a small LIBERO rollout evaluation using release dataset stats, same env
   setup, and one-step inference. This checks whether action distribution and
   normalization remain usable under rollout.
3. Only start longer/full training if both latency and rollout quality are
   acceptable. Training loss alone is not enough to promote the method.

Current 0,1,2,3 retry:

```text
RUN_ID=meanflow_c_action_fd_bf16_0123_2k_20260516
task=libero_one_step_meanflow_2cam224_1e-4
max_steps=2000
mixed_precision=bf16
gpus=0,1,2,3
```

The 20-step smoke on the same cards completed and saved:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_smoke3_20260516/checkpoints/weights/step_000020.pt
```

## C Pilot 2k Gate Result

Archived on 2026-05-16.

The C-action-only finite-difference bf16 2k pilot completed:

```text
run:  runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_2k_20260516
ckpt: runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_2k_20260516/checkpoints/weights/step_002000.pt
step_002000 loss=0.1148
step_002000 loss_meanflow_target=0.1148
speed ~= 0.22 step/s
```

Latency benchmark:

```text
output: evaluate_results/latency/meanflow_c_action_fd_bf16_0123_2k_1_vs_10_20260516.json
gpu:    NVIDIA H200, gpu0
warmup: 5
repeats: 30

steps=1:
  end-to-end mean 110.996 ms, median 91.458 ms
  denoise-loop mean 29.803 ms

steps=10:
  end-to-end mean 397.052 ms, median 365.379 ms
  denoise-loop mean 350.643 ms

ratio:
  end-to-end 10/1 mean = 3.58x
  denoise-loop 10/1 mean = 11.77x
```

Small LIBERO rollout:

```text
task_set:        libero_gap_probe_v1
output:          evaluate_results/libero/meanflow_c_fd_0123_2k_gap_probe_t5_20260516
num_trials:      5 per task
inference_steps: 1
dataset_stats:   checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
gpu:             0
```

Results:

```text
libero_goal    task6  0/5  put the cream cheese in the bowl
libero_goal    task5  0/5  push the plate to the front of the stove
libero_10      task0  0/5  put both the alphabet soup and the tomato sauce in the basket
libero_10      task2  0/5  turn on the stove and put the moka pot on it
libero_spatial task3  4/5  pick up the black bowl on the cookie box and place it on the plate
libero_10      task8  0/5  put both moka pots on the stove
libero_10      task3  0/5  put the black bowl in the bottom drawer of the cabinet and close it

total: 4/35 = 11.43%
```

Interpretation:

```text
Latency gate passes: one-step inference is substantially faster than 10-step
inside the denoise loop, and still gives a 3.58x end-to-end mean speedup in this
synthetic benchmark.

Rollout gate does not pass: total success is 11.43%, and the LIBERO-10 subset is
0/20. The one spatial task at 4/5 shows the checkpoint is not completely broken,
but quality is far below the release baseline for this probe.

Decision: do not start full C-action-only training from this 2k result alone.
Next step should be diagnostic before promotion: inspect failure videos/actions
or run D0/D1/C-joint only if we want to test whether the failure is due to
action-only MeanFlow capacity, missing joint video/action training, or short
2k convergence.
```

## Action Trace Diagnostic Rerun

Archived on 2026-05-16.

After adding rollout video and executed-action trace saving, the selected
trial-0 probe was rerun:

```text
output: evaluate_results/libero/meanflow_c_fd_0123_2k_gap_probe_trace_t0_20260516
ckpt:   runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_2k_20260516/checkpoints/weights/step_002000.pt
stats:  checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
steps:  1 inference step
trials: trial 0 only for each selected task
```

Result:

```text
overall:        1/7 = 14.29%
libero_spatial: 1/1
libero_goal:    0/2
libero_10:      0/4
```

Per-task result:

```text
libero_spatial task3  1/1  pick up the black bowl on the cookie box and place it on the plate
libero_goal    task5  0/1  push the plate to the front of the stove
libero_goal    task6  0/1  put the cream cheese in the bowl
libero_10      task0  0/1  put both the alphabet soup and the tomato sauce in the basket
libero_10      task2  0/1  turn on the stove and put the moka pot on it
libero_10      task3  0/1  put the black bowl in the bottom drawer of the cabinet and close it
libero_10      task8  0/1  put both moka pots on the stove
```

Trace summary:

```text
suite          task  success  actions  replans  max_pos  max_rot  pos>=0.90  gripper
libero_10        0   false      700      70      0.564    0.146    0.000%     -1:594, +1:106, transitions=26
libero_10        2   false      700      70      0.802    0.285    0.000%     -1:370, +1:330, transitions=51
libero_10        3   false      700      70      0.696    0.305    0.000%     -1:700, transitions=0
libero_10        8   false      700      70      0.813    0.286    0.000%     -1:693, +1:7, transitions=10
libero_goal      5   false      400      40      1.326    0.109    1.250%     -1:394, +1:6, transitions=4
libero_goal      6   false      400      40      0.945    0.127    0.250%     -1:352, 0:1, +1:47, transitions=25
libero_spatial   3   true        82       9      1.296    0.122    6.504%     -1:48, +1:34, transitions=1
```

Interpretation:

```text
The failure is not a simple global action-scale explosion. The successful
spatial trace has one of the largest translational maxima and the highest
near-limit fraction, while several failures stay well inside the release-scale
range.

The strongest action-level symptom is gripper/timing semantics. The successful
spatial trace has a clean one-transition gripper pattern: open for 48 executed
actions, then close for 34. In contrast, LIBERO-10 task3 never closes at all,
task8 almost never closes, and task0/task2 show many open/close transitions.

This supports the current gate decision: do not promote C-action-only from this
2k checkpoint. The checkpoint has a usable local motion prior for a simple
spatial task, but it does not yet produce reliable long-horizon task/gripper
structure for LIBERO-10.
```

## Matched Release Trial-0 Rerun

Archived on 2026-05-16.

To check whether the selected trial-0 states are intrinsically solvable, the
same 7 selected tasks and the same `trial0` were rerun with release-style
multi-step inference:

```text
meanflow output: evaluate_results/libero/meanflow_c_fd_0123_2k_gap_probe_trace_t0_20260516
release_1 output: evaluate_results/libero/release1_gap_probe_trace_t0_20260516
release_10 output: evaluate_results/libero/release10_gap_probe_trace_t0_20260516
```

Each release output contains videos and action traces:

```text
<output>/<suite>/videos/*.mp4
<output>/<suite>/action_traces/*_action_trace.json
```

Matched success result:

```text
model                  overall  libero_goal  libero_spatial  libero_10
MeanFlow C 2k 1-step   1/7      0/2          1/1             0/4
release_1              6/7      2/2          1/1             3/4
release_10             7/7      2/2          1/1             4/4
```

Per-task result:

```text
suite          task  MeanFlow C  release_1  release_10  description
libero_goal      6   0/1         1/1        1/1         put the cream cheese in the bowl
libero_10        0   0/1         1/1        1/1         put both the alphabet soup and the tomato sauce in the basket
libero_goal      5   0/1         1/1        1/1         push the plate to the front of the stove
libero_10        2   0/1         0/1        1/1         turn on the stove and put the moka pot on it
libero_spatial   3   1/1         1/1        1/1         pick up the black bowl on the cookie box and place it on the plate
libero_10        8   0/1         1/1        1/1         put both moka pots on the stove
libero_10        3   0/1         1/1        1/1         put the black bowl in the bottom drawer of the cabinet and close it
```

Key gripper trace examples:

```text
MeanFlow spatial task3 success:
  gripper segments: 0-47 open, 48-81 close

release_10 spatial task3 success:
  gripper segments: 0-38 open, 39-80 close

release_10 task2 success:
  gripper segments: 0-40 open, 41-94 close, 95-178 open, 179-225 close

MeanFlow libero_10 task3 failure:
  gripper segments: 0-699 open only

MeanFlow libero_10 task2 failure:
  52 gripper segments/transitions over the full 700 executed actions
```

Interpretation:

```text
The matched release rerun rules out "bad selected initial states" as the main
explanation. release_10 solved all 7 exact trial-0 probes, and release_1 solved
6/7.

The one release_1 miss, libero_10 task2, is useful: it is exactly the kind of
longer multi-stage manipulation where release_10's additional denoising quality
matters. MeanFlow also failed it, but with unstable gripper timing rather than a
simple action-scale blowup.

The dominant MeanFlow failure mode remains task/gripper timing. Successful
release traces show structured open/close phases tied to subgoals, while the 2k
C-action-only MeanFlow checkpoint often either never closes, almost never
closes, or toggles the gripper repeatedly through a full-horizon failure.

Decision remains unchanged: do not promote the 2k C-action-only checkpoint to a
full training run based on latency alone. The next defensible step is a
diagnostic objective or training variant that tests whether the failure is
caused by action-only fine-tuning, missing joint video/action training, or weak
gripper/timing supervision.
```

## Gripper/Timing Diagnostic Plan

Archived on 2026-05-16.

The next work should separate three possible causes:

```text
H1: the 2k C-action-only checkpoint mostly fails because gripper timing is bad.
H2: 2k steps are simply not enough for the MeanFlow objective to converge.
H3: action-only MeanFlow fine-tuning breaks FastWAM's original joint video/action structure.
```

Execution order:

```text
1. No-training gripper/timing diagnostics.
2. C-action-only continuation to 6k/8k if diagnostics do not fully explain the failure.
3. A gripper-weighted action-only MeanFlow diagnostic if raw traces show weak gripper supervision.
4. C-joint only after the above, because it requires a joint training loss and longer training.
```

Phase 1, no-training diagnostics:

```text
Add raw action trace stages:
  model normalized action
  denormalized dataset action
  gripper scaled to [-1, +1] before LIBERO sign inversion
  continuous LIBERO action before optional sign/binning
  final action sent to env

Summarize each trace:
  first_close_step
  close_ratio
  gripper transition count
  gripper segments
  max_pos / max_rot
  episode length

Run hybrid replay on the same trial0 states:
  A: MeanFlow original action
  B: release10 original action
  C: MeanFlow xyz/rot + release10 gripper
  D: release10 xyz/rot + MeanFlow gripper
  E: MeanFlow xyz/rot + smoothed MeanFlow gripper
```

Interpretation gate:

```text
C improves strongly over A:
  gripper/timing is a main cause.

D drops strongly from B:
  MeanFlow gripper alone is enough to break otherwise-good release motion.

E improves over A:
  the issue may be gripper threshold/jitter and not only representation learning.

C still fails:
  xyz/rot motion timing or subgoal sequencing is also broken.
```

Implementation note from 2026-05-16:

```text
The new raw-trace and replay tooling is in place.

Observed replay sanity on original traces:
  release_original and meanflow_original both reproduce the same broad success/failure trend
  when replayed with the correct trace-config num_steps_wait.

One caveat remains:
  replay uses LIBERO stateful env resets, so each variant must deep-copy the initial state
  before set_init_state. The hybrid replay script was updated for this.

The replay path is therefore usable for small hybrid checks, but it is slower than the trace
summaries. Treat it as a targeted diagnostic, not as the main evaluation path.
```

Hybrid replay result on 2026-05-16:

```text
output: evaluate_results/libero/gripper_timing_hybrid_replay_t0_20260516_focus
trial:  trial0
tasks:  libero_gap_probe_v1, 7 tasks

variant                            success
meanflow_original                  1/7
release_original                   7/7
meanflow_motion_release_gripper    0/7
release_motion_meanflow_gripper    1/7
```

Interpretation:

```text
The result rules out a simple "only gripper is bad" explanation.

If MeanFlow's xyz/rot motion were mostly good and only gripper were bad, then
meanflow_motion_release_gripper should have rescued several tasks. It rescued
none: 0/7.

If MeanFlow's gripper were harmless, release_motion_meanflow_gripper should
have stayed close to release_original. It collapsed from 7/7 to 1/7.

Therefore the failure is two-sided:
  1. MeanFlow gripper/timing is bad enough to break otherwise-good release motion.
  2. MeanFlow motion/subgoal timing is not recoverable by simply grafting release gripper.

The successful spatial task also shows that gripper and motion are coupled in
time: both original MeanFlow and original release succeed, but swapping the
gripper sequence between them fails. This means the gripper should not be
treated as an independent post-hoc signal; it must be learned in phase with the
motion trajectory.

Main conclusion:
  MeanFlow C 2k is worse than release because the action policy lost coupled
  long-horizon action timing, not merely because of a bad gripper threshold.
  The evidence points more toward action-only MeanFlow disrupting FastWAM's
  learned joint action structure, or the MeanFlow target being underconstrained
  for task-phase semantics, than toward a one-dimensional gripper bug.
```

Phase 2, convergence test:

```text
Continue C-action-only finite-difference MeanFlow from step_002000 to 6k or 8k.

If rollout and gripper segments improve materially, 2k was undertrained.
If loss improves but rollout/gripper does not, the failure is not just short training.
```

Phase 3, gripper-targeted training diagnostic:

```text
C-gripper-weight:
  load release checkpoint
  keep action-only MeanFlow
  keep video frozen
  increase the gripper action dimension loss weight, e.g. 5x or 10x
  run a 2k pilot

This is a diagnostic, not a final paper-pure objective.
```

Phase 4, C-joint:

```text
load release checkpoint
unfreeze video expert + action expert + proprio encoder + MeanFlow conditioning
video branch: original FastWAM video flow loss
action branch: MeanFlow action loss
use the same mixed-attention MoT pass so video loss constrains the joint representation
```

Expected answer from C-joint:

```text
C-joint improves over C-action-only:
  action-only fine-tuning likely breaks or underuses the coupled video/action representation.

C-joint also fails:
  focus on the MeanFlow target/formulation or gripper-specific supervision.
```
