# T1 Force — Soft Landing Experiment

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TianyuQ/robot_feet.git
cd robot_feet
git checkout clean_t1_force
uv sync --extra learning
```

For GPU clusters with non-standard CUDA:
```bash
uv pip install "jax[cuda12]"
```

Log in to W&B before training:
```bash
uv run wandb login
```

## Experiment

Three runs on separate GPUs. The only differences are the soft landing reward scale and whether force sensor readings are in the observation.

```bash
# Run 1 — Baseline: no force in reward or observations
CUDA_VISIBLE_DEVICES=0 uv run python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --run_render=False \
  --observe_force_sensors=False \
  --suffix=baseline

# Run 2 — Force in reward only: soft landing penalty, no force in observations
CUDA_VISIBLE_DEVICES=1 uv run python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --run_render=False \
  --soft_landing_scale=-1e-5 \
  --observe_force_sensors=False \
  --suffix=force_reward_only

# Run 3 — Force in reward and observations
CUDA_VISIBLE_DEVICES=2 uv run python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --run_render=False \
  --soft_landing_scale=-1e-5 \
  --observe_force_sensors=True \
  --suffix=force_reward_and_obs
```
