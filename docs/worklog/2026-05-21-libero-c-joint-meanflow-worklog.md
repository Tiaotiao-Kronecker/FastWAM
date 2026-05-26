# 2026-05-21 LIBERO C-joint MeanFlow Worklog

## Purpose

Implement the originally planned C-joint MeanFlow variant:

```text
load release checkpoint
unfreeze video expert + action expert + proprio encoder + MeanFlow conditioning
video branch: original FastWAM video flow loss
action branch: paper-style MeanFlow loss
```

This is intentionally different from the existing joint finite-difference
MeanFlow pilot. The old `FastWAMJointMeanFlow` changes both video and action to
finite-difference MeanFlow. This C-joint implementation keeps the video
objective as original FastWAM flow matching and changes only the action branch
to MeanFlow, while training through one shared video/action MoT forward.

## Files Added

```text
src/fastwam/models/wan22/fastwam_c_joint_meanflow.py
configs/model/fastwam_c_joint_meanflow.yaml
configs/task/libero_c_joint_meanflow_2cam224_1e-4.yaml
```

Runtime factory added:

```text
fastwam.runtime.create_fastwam_c_joint_meanflow
```

## Difference From Existing Variants

```text
C-action-only:
  file: fastwam_one_step_meanflow.py
  video: frozen first-frame context only
  action: MeanFlow
  train signal: action-only

joint finite-difference MeanFlow:
  file: fastwam_joint_meanflow.py
  video: finite-difference MeanFlow
  action: finite-difference MeanFlow
  train signal: joint, but video objective is no longer original FastWAM

C-joint MeanFlow:
  file: fastwam_c_joint_meanflow.py
  video: original FastWAM flow loss
  action: MeanFlow, default paper_jvp
  mask: FastWAM first-frame action mask, not action-to-full-video mask
  train signal: video flow anchor plus release-style first-frame action policy
```

The C-joint hypothesis is narrower than "joint MeanFlow works":

```text
Does adding the original video flow anchor, while keeping the action policy mask
as close to FastWAM release as possible, stabilize action MeanFlow and recover
long-horizon action timing that C-action-only lost?
```

## Original Flow Matching Variable Flow

For clean data `x_0` and Gaussian noise `epsilon`, FastWAM uses:

```text
x_sigma = (1 - sigma) * data + sigma * noise
target velocity v = noise - data
```

Training samples are generated as:

```text
dataset video/action
  -> video is encoded by VAE
  -> sample noise
  -> sample random timestep/sigma
  -> x_sigma = (1 - sigma) * data + sigma * noise
  -> target = noise - data
  -> model predicts target velocity
```

Code mapping:

```text
data:
  video branch  -> input_latents
  action branch -> action

noise:
  video branch  -> noise_video
  action branch -> noise_action

sigma/timestep:
  video branch  -> timestep_video
  action branch -> timestep_action

x_sigma:
  video branch  -> latents / noisy_video
  action branch -> noisy_action

target:
  video branch  -> target_video = noise_video - input_latents
  action branch -> target_action = noise_action - action
```

Important scheduler methods:

```text
sample_training_t(): random training timestep sampling
add_noise():         (1 - sigma) * sample + sigma * noise
training_target():   noise - sample
step():              sample + model_output * delta
```

Training uses random timestep sampling. Inference uses a fixed schedule from
high noise to low noise. Since inference deltas are negative, `step()` is
equivalent to:

```text
x_{sigma-d} = x_sigma - d * v_pred
```

## C-joint Training Flow

The C-joint implementation builds original video flow states first:

```text
video:
  input_latents = VAE(video)
  noise_video = randn_like(input_latents)
  timestep_video = sample_training_t()
  noisy_video = add_noise(input_latents, noise_video, timestep_video)
  target_video = noise_video - input_latents
```

The attention mask is intentionally the base FastWAM mask:

```text
video tokens -> video tokens, following video_expert.build_video_to_video_mask()
action tokens -> action tokens
action tokens -> first-frame video tokens only
action tokens -> not allowed to see future video latent tokens
```

This was updated after implementation review. The first draft inherited the
`FastWAMJoint` full-video action mask; the current code overrides
`_build_mot_attention_mask()` and delegates to `FastWAM._build_mot_attention_mask()`
so training-time action visibility matches release-style action inference.

Then action MeanFlow states:

```text
action:
  data = action
  noise_action = randn_like(action)
  sample r,t as sigma_start,sigma_end with r <= t
  timestep_start = r * num_train_timesteps
  timestep_end = t * num_train_timesteps
  noisy_action = add_noise(action, noise_action, timestep_end)
  target_action_velocity = noise_action - action
```

The model function is:

```text
u_theta(x_t, r, t)
```

Implementation detail:

```text
t enters the normal action timestep embedding.
r enters as MeanFlow start conditioning:
  action_pre["t_mod"] = action_pre["t_mod"] + g(r)
```

This start conditioning is necessary but is not the whole MeanFlow method.
MeanFlow also changes the objective:

```text
meanflow_target = v - (t - r) * du_theta/dt
```

For the default `paper_jvp` objective, `du_theta/dt` is computed with
`torch.func.jvp` along:

```text
dx_t/dt = target_action_velocity
dt/dt = 1
dr/dt = 0
```

The final C-joint loss is:

```text
loss = lambda_video * original_video_flow_loss
     + lambda_meanflow_action * MSE(u_theta, stopgrad(meanflow_target))
```

Optional diagnostic action velocity/endpoint losses are wired in the class but
default to zero in config.

## Inference Flow

Current C-joint action evaluation intentionally uses release-style action
policy inference, not joint future-video generation:

```text
input current image
  -> encode first frame to video latent
  -> prefill video KV cache
  -> sample random action noise
  -> denoise action through action scheduler
```

At each action denoising step, the action path applies MeanFlow start
conditioning with fixed `r = 0`:

```text
u_theta(x_t, 0, t)
```

For the intended one-step evaluation:

```text
x_1 = random action noise
u = u_theta(x_1, 0, 1)
x_0 = x_1 - u
```

This is why the implementation overrides `infer_action()` to call the original
FastWAM first-frame action policy path. Without that override, inheriting from
`FastWAMJoint` would expose `num_video_frames` in the signature and LIBERO eval
could route through future-video joint inference, making results less directly
comparable to prior one-step action evaluations.

After the mask update, train/infer visibility is now aligned for the action
path:

```text
training action path: noisy action + first-frame video tokens
inference action path: noisy action + first-frame video tokens
```

The video branch is still trained with original video flow loss, but action
tokens do not use future video latents as privileged inputs.

Important caveat:

```text
This implementation is theoretically aligned primarily with num_inference_steps=1.
If num_inference_steps > 1, current inference keeps r=0 for every step, so it is
not a clean local flow solver over sub-intervals.
```

## Default Config

```text
task: libero_c_joint_meanflow_2cam224_1e-4
objective: paper_jvp
equal_time_prob: 0.25
trainable_scope: joint
train_proprio_encoder: true
lambda_video: 1.0
lambda_meanflow_action: 1.0
lambda_action_velocity: 0.0
lambda_action_endpoint: 0.0
resume: checkpoints/fastwam_release/libero_uncond_2cam224.pt
save_training_state: true
```

The class also supports:

```text
model.c_joint_meanflow.objective=finite_difference
```

This fallback is for diagnostics and is not the default original C-joint plan.

## Verification

Static compile passed:

```bash
.conda/fastwam/bin/python -m py_compile \
  src/fastwam/models/wan22/fastwam_c_joint_meanflow.py \
  src/fastwam/runtime.py
```

Hydra config resolution passed:

```bash
.conda/fastwam/bin/python scripts/train.py \
  task=libero_c_joint_meanflow_2cam224_1e-4 --cfg job
```

Factory import and `infer_action` signature check passed:

```bash
.conda/fastwam/bin/python -c \
  "import inspect; from fastwam.models.wan22.fastwam_c_joint_meanflow import FastWAMCJointMeanFlow; print(inspect.signature(FastWAMCJointMeanFlow.infer_action))"
```

A full `max_steps=1` smoke was not run in this archive because the trainer saves
a large final weight checkpoint when max steps is reached.

## Suggested Run

For the default paper-JVP path, prefer fp32/no mixed precision:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_zero2.sh 4 \
  task=libero_c_joint_meanflow_2cam224_1e-4 \
  mixed_precision=no
```

For a lower-risk finite-difference diagnostic run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_zero2.sh 4 \
  task=libero_c_joint_meanflow_2cam224_1e-4 \
  model.c_joint_meanflow.objective=finite_difference
```

Evaluation should prioritize `num_inference_steps=1`, because that is the
MeanFlow one-step use case this implementation is designed to test.

## Current C-joint Audit

Reviewed on 2026-05-21 after the action-mask alignment change.

Confirmed alignment with base FastWAM:

```text
action visibility:
  current C-joint delegates `_build_mot_attention_mask()` to base FastWAM
  base FastWAM mask allows:
    video -> video, following video_expert.build_video_to_video_mask()
    action -> action
    action -> first-frame video tokens only

video objective:
  current C-joint still uses original video flow matching:
    noisy_video = add_noise(input_latents, noise_video, timestep_video)
    target_video = noise_video - input_latents
```

Important differences from base FastWAM:

```text
action objective:
  base FastWAM:
    predicts instantaneous flow target noise_action - action

  current C-joint:
    predicts MeanFlow action velocity u_theta(x_t, r, t)
    default objective is paper_jvp
    meanflow_target = v - (t - r) * du_theta/dt

conditioning:
  current C-joint adds action_expert.meanflow_start_embedding
  and action_expert.meanflow_start_projection, then injects g(r)
  into action_pre["t_mod"].

inference:
  current C-joint action inference uses the base FastWAM first-frame
  action-policy path, but `_predict_action_noise_with_cache()` applies
  MeanFlow start conditioning with fixed r = 0.
```

Potential implementation risks:

```text
1. Multi-step action inference is not theoretically clean.
   Current inference uses r = 0 for every action denoising step.
   This matches the one-step endpoint use case, but for num_inference_steps > 1
   a local interval solver would likely need step-dependent r.

2. MeanFlow r,t sampling bypasses scheduler shift.
   Base FastWAM action training uses train_action_scheduler.sample_training_t(),
   which applies train_shift. Current C-joint samples raw uniform sigma pairs
   with torch.rand() and then maps them to timesteps. This changes the action
   training-time distribution.

3. C-joint is not full future-video-conditioned action training.
   After the mask alignment, action tokens do not see future video latent tokens.
   With video_dit_config.action_conditioned=false, video loss mainly anchors the
   video branch and joint execution context; it should not be interpreted as
   direct future-video privileged supervision for the action expert.

4. The class still inherits FastWAMJoint.
   The action mask and infer_action path are overridden back to FastWAM-style
   behavior, but inherited infer_joint/model.infer behavior can still expose
   a future-video joint generation path. LIBERO action evaluation should call
   infer_action explicitly.

5. paper_jvp should be run in fp32/no mixed precision when possible.
   The default training config is bf16, but the JVP path is more numerically
   sensitive than the original FastWAM flow-matching loss.
```

Action-only visibility check:

```text
Previous action-only MeanFlow training also uses first-frame-only video context.

FastWAMOneStepMeanFlow inherits FastWAMOneStepAction and does not override the
base FastWAM MoT mask. Its training builds `video_pre` from first_frame_latents
only:

  video_pre = video_expert.pre_dit(x=first_frame_latents, ...)

Then `_predict_meanflow_action_velocity()` runs the inherited base FastWAM mask.
Because the video sequence contains only first-frame tokens, action tokens can
attend to action tokens plus that first-frame video token sequence only.
They do not see future video latent tokens.
```
