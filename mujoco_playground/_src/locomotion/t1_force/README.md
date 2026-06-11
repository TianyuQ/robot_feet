# T1 Force — Soft Landing Experiment

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd robot_feet
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
