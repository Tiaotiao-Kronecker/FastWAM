# 2026-05-18 LIBERO Joint Finite-Difference MeanFlow Pilot

## Goal

Build the smallest FastWAM-faithful MeanFlow pilot that keeps the original
joint video/action structure, while avoiding the unstable paper-JVP path.

The pilot should answer one question:

```text
If video and action are trained jointly under a finite-difference MeanFlow
objective, does the coupled model behave more like release than the current
action-only MeanFlow pilot?
```

## Design Principles

```text
1. Keep FastWAM's causal structure.
2. Keep video/action joint training.
3. Use finite difference only.
4. Minimize new code paths.
5. Reuse the existing release checkpoint and LIBERO eval chain.
```

## What This Pilot Will Do

```text
load release checkpoint
use FastWAMJoint attention topology
train video and action jointly
replace the action-only loss with a joint finite-difference MeanFlow loss
keep the first-frame causal video mask
keep the same LIBERO data normalization
```

## What It Will Not Do

```text
no paper-JVP in the pilot
no new trainer logic
no new eval protocol
no full-horizon scaling before a smoke pass
```

## Implementation Scope

```text
1. Add a joint MeanFlow model class.
2. Add a runtime factory for that class.
3. Add a config file for LIBERO 2cam.
4. Add a pilot task config.
5. Smoke-test forward loss and checkpoint loading.
6. Run a short training pilot only after the smoke passes.
```

## Validation Gates

```text
smoke:
  model instantiates
  loss forward passes
  release checkpoint loads

pilot:
  2k steps only
  evaluate with the existing gap-probe chain
  promote only if rollout and trace behavior improve
```

## Expected Outcome

```text
If this pilot improves over action-only MeanFlow, the next step is a larger
joint run. If it does not, we stop and diagnose the objective instead of
scaling further.
```
