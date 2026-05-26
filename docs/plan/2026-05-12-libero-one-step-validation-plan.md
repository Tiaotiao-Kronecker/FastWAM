# 2026-05-12 LIBERO One-Step Validation Plan

## Goal

Use the official Fast-WAM LIBERO release checkpoint as the baseline anchor, then evaluate whether one-step fine-tuning objectives improve over the same checkpoint with `num_inference_steps=1`.

The baseline checkpoint is:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

The matching dataset statistics file is:

```text
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

We do not need to fine-tune a new Fast-WAM baseline before comparing one-step methods. The release checkpoint is the trained baseline weight released by the authors.

## Evaluation Anchor

First reproduce the paper-style LIBERO release result:

- suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- tasks: all 40 LIBERO tasks
- trials: 50 per task
- inference steps: 10
- text CFG scale: 1.0
- checkpoint: official LIBERO release checkpoint

Then run the same release checkpoint with fewer inference steps:

```text
release_1
release_2
release_4
release_10
```

This curve tells us how much headroom one-step fine-tuning has. A one-step method is useful only if it beats `release_1` and preferably approaches `release_4` or `release_10`.

## One-Step Methods

All one-step methods should initialize from the release checkpoint, use the same LIBERO data, and use real task instructions. `override_instruction` is allowed only for smoke tests, never for formal comparison.

### Shortcut

Shortcut trains a velocity field conditioned on the step size `d`.

At one-step inference:

```text
x_1 = noise
pred_action = x_1 - v_big
```

Current loss:

```text
L_velocity = MSE(v_big, noise - action)
L_endpoint = MSE(x_1 - v_big, action)
L_consistency = MSE(v_big, stopgrad(0.5 * (v_half_1 + v_half_2)))
L_half_velocity = 0.5 * MSE(v_half_1, noise - action)
                + 0.5 * MSE(v_half_2, noise - action)

L_shortcut = 0.25 * L_velocity
           + 0.25 * L_endpoint
           + 0.25 * L_consistency
           + 0.25 * L_half_velocity
```

Risk: if `sigma=1` and `d=1` are fixed, endpoint and velocity losses are mathematically equivalent. A stronger formal run should randomize `sigma` and `d`, subject to `d <= sigma`.

### MeanFlow

MeanFlow trains the model to predict an interval-mean velocity, not only an instantaneous velocity.

For one-step inference, the interval is from `sigma=1` to `sigma=0`:

```text
pred_action = x_1 - u_mean
```

Current loss:

```text
u = model(x_t, timestep_end=t, timestep_start=r)
x_prev = x_t - eps * (noise - action)
u_prev = model(x_prev, t - eps, r)
du_dt = stopgrad((u - u_prev) / eps)

meanflow_target = (noise - action) - (t - r) * du_dt

L_meanflow_target = MSE(u, meanflow_target)
L_velocity = MSE(u, noise - action)
L_endpoint = MSE(x_t - (t - r) * u, action)

L_meanflow = 0.5 * L_meanflow_target
           + 0.25 * L_velocity
           + 0.25 * L_endpoint
```

Risk: finite-difference targets can be numerically sensitive. We should start from smoke and pilot runs before spending full 20k steps.

## Execution Order

1. Run one-task LIBERO release smoke at `num_inference_steps=10`.
2. If smoke passes, run paper-style full `release_10`.
3. Run release step curve: `release_1`, `release_2`, `release_4`, `release_10`.
4. Add LIBERO one-step task configs for shortcut and meanflow.
5. Run zero-update sanity checks: one-step wrappers loaded from release checkpoint should initially match `release_1` closely if added conditioning layers are zero-initialized.
6. Run 1-step smoke training.
7. Run pilot fine-tuning at 2k/5k steps.
8. Run formal fine-tuning at 20k steps only for methods that survive pilot checks.

## Formal Comparison Table

```text
release_1
release_2
release_4
release_10
shortcut_1
meanflow_1
```

All rows must use the same LIBERO suites, trials, dataset stats, text CFG scale, action horizon, and replan settings.
