# T1 Force — Soft Landing Experiment

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
