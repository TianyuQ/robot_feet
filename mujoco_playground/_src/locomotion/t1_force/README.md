# T1 Force — Locomotion with Foot Force Sensors

This module adds foot force sensors to the T1 humanoid robot. The force sensor
readings can be included in the policy observation or used only as an auxiliary
reward signal.

## Environments

| Name | Description |
|------|-------------|
| `T1ForceJoystickFlatTerrain` | Flat terrain, force sensors in observation + reward |
| `T1ForceJoystickRoughTerrain` | Rough terrain, force sensors in observation + reward |
| `T1ForceJoystickFlatTerrainForceRewardOnly` | Flat terrain, force used as reward signal only (not observed) |

## Training

From the repo root, with wandb logging:

```bash
python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb
```

To disable domain randomization or override timesteps:

```bash
python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --nodomain_randomization \
  --num_timesteps=100000000
```

Checkpoints and logs are saved to `learning/logs/`.

## Soft landing experiment

To compare the effect of the `soft_landing` reward, run these two commands (on
separate GPUs if available):

```bash
# Baseline — no soft landing penalty
CUDA_VISIBLE_DEVICES=0 python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --suffix=baseline

# With soft landing penalty
CUDA_VISIBLE_DEVICES=1 python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --soft_landing_scale=-1e-5 \
  --suffix=soft_landing
```

`CUDA_VISIBLE_DEVICES` pins each run to a specific GPU. The `--suffix` flag
keeps the W&B runs and checkpoint directories distinct.

## Evaluation

To visualize a trained policy without further training:

```bash
python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --play_only \
  --load_checkpoint_path=learning/logs/<run_name>/checkpoints/
```

## What the sensors add

Each foot has a 3-axis force sensor (`left_foot_force`, `right_foot_force`).
The sensor readings are included in both the policy observation (`state`) and
the privileged critic observation (`privileged_state`).

The `soft_landing` reward penalizes high contact force magnitude at the instant
a foot first touches the ground. It is zero by default and enabled via
`--soft_landing_scale`.
