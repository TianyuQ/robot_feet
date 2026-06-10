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
The environment adds three force-related reward terms on top of the standard T1
locomotion reward:

- **`force_sensor_balance`** — penalizes asymmetric loading between feet
- **`force_sensor_stability`** — penalizes rapid changes in foot force
- **`force_impulse_penalty`** — penalizes high-impact contact forces

Set `observe_force_sensors: false` in the env config to use the `ForceRewardOnly`
variant behavior without switching environments.
