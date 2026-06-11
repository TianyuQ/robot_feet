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

Run from the repo root. Baseline and soft landing on separate GPUs:

```bash
# Baseline — no soft landing penalty
CUDA_VISIBLE_DEVICES=0 uv run python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --run_render=False \
  --suffix=baseline

# With soft landing penalty
CUDA_VISIBLE_DEVICES=1 uv run python learning/train_jax_ppo.py \
  --env_name=T1ForceJoystickFlatTerrain \
  --use_wandb \
  --soft_landing_scale=-1e-5 \
  --run_render=False \
  --suffix=soft_landing
```
