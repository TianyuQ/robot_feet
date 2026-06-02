# robot_feet

Research codebase for **Booster T1 humanoid locomotion** with **foot force sensors**, built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (GPU-accelerated MJX) and optional [Booster Gym](https://github.com/boosterrobotics/booster_gym) (Isaac Gym) pipelines.

The main focus is learning joystick-controlled walking policies that use (or are regularized by) bilateral foot contact forces, then evaluating and visualizing force profiles, impulses, and rollout videos.

---

## Repository layout

```
robot_feet/                          # workspace root (this README)
├── robot_feet/                      # MuJoCo Playground fork + T1 / T1_Force extensions
│   ├── mujoco_playground/           # simulation environments (MJX)
│   │   └── _src/locomotion/
│   │       ├── t1/                  # baseline T1 (no force obs in policy)
│   │       └── t1_force/            # T1 with foot force sensors + force_impulse_penalty
│   ├── learning/
│   │   ├── train_jax_ppo.py         # generic Brax PPO entrypoint (any registered env)
│   │   └── scripts/                 # T1-specific train / eval / plot scripts (start here)
│   └── pyproject.toml
├── booster_gym/                     # Isaac Gym training / play / deploy for T1
└── isaacgym/                        # local Isaac Gym install (not part of this repo)
```

| Component | Role |
|-----------|------|
| **`robot_feet/mujoco_playground`** | JAX/MJX environments, rewards, domain randomization |
| **`robot_feet/learning/scripts`** | End-to-end CLI: train → checkpoint → eval → CSV/plots/video |
| **`booster_gym`** | Alternative Isaac Gym RL stack; sim-to-real export and deploy |
| **`robot_feet/learning/train_jax_ppo.py`** | Upstream-style PPO trainer for any Playground env name |

---

## MuJoCo environments (T1 family)

Registered in `mujoco_playground/_src/locomotion/__init__.py`:

| Env name | Robot | Force in obs? | `force_impulse_penalty` | Terrain |
|----------|-------|---------------|-------------------------|---------|
| `T1JoystickFlatTerrain` | T1 | No | No | Flat |
| `T1JoystickRoughTerrain` | T1 | No | No | Rough |
| `T1ForceJoystickFlatTerrain` | T1_Force | Yes (`observe_force_sensors=True`) | Yes (default scale **-0.1**) | Flat |
| `T1ForceJoystickFlatTerrainForceRewardOnly` | T1_Force | **No** (forces only shape reward) | Yes | Flat |
| `T1ForceJoystickRoughTerrain` | T1_Force | Yes | Yes | Rough |

**Joystick task:** sample random `(vx, vy, vyaw)` commands; policy tracks velocity while staying upright. Force sensors report 6D wrench per foot (used in observations and/or `force_impulse_penalty`).

**Key reward (force runs):** `force_impulse_penalty` penalizes large vertical foot forces (impulse-style term). Override at train/eval time with `--force-impulse-penalty-scale` (e.g. `0` to disable).

**Training budget (PPO):** 200M environment steps, eval every ~10M steps (`locomotion_params.py`).

---

## Installation (MuJoCo / MJX path — recommended)

Requirements: **Python ≥ 3.10**, NVIDIA GPU with **CUDA 12** JAX recommended.

```bash
cd robot_feet/robot_feet

# virtualenv (example)
python3.11 -m venv .venv
source .venv/bin/activate

# JAX with GPU
pip install -U "jax[cuda12]"

# Playground + dependencies from source
pip install -e ".[all]"

# Script extras (TensorBoard, pandas, etc.)
pip install -r learning/scripts/requirements.txt
pip install tensorboardX pandas cloudpickle

# Verify GPU backend
python -c "import jax; print(jax.default_backend())"   # expect: gpu

# Verify env import (downloads Menagerie assets on first run)
python -c "import mujoco_playground"
```

**Reproducibility (Ampere / RTX 30–40):** for stable RL training,

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

**Rendering:** evaluation videos use `mediapy` (included in `pyproject.toml`). Headless servers may need `export MUJOCO_GL=egl`.

---

## Quick start: T1 with force sensors

All commands below assume:

```bash
cd robot_feet/robot_feet/learning/scripts
```

### 1. Training

```bash
# Full training run (200M steps) — checkpoints + TensorBoard under logs/
python t1_joystick_flat_terrain_force.py --train

# Disable TensorBoard
python t1_joystick_flat_terrain_force.py --train --no-tensorboard

# Softer / no force impulse penalty
python t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale -0.3
python t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale 0
```

**Experiment directory naming:**

```
logs/T1ForceJoystickFlatTerrain-Fimp{scale}-steps{num_timesteps}/
├── checkpoints/000123456789/     # Brax/Orbax checkpoints
├── events.out.tfevents.*         # TensorBoard (if enabled)
└── ...                           # eval outputs (see below)
```

Monitor training:

```bash
tensorboard --logdir logs
```

See also: [`learning/scripts/README_TensorBoard.md`](robot_feet/learning/scripts/README_TensorBoard.md).

### 2. Evaluation (rollout + metrics)

Load a checkpoint and run one deterministic episode at a fixed command:

```bash
python t1_joystick_flat_terrain_force.py --eval \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000/checkpoints \
  --x-vel 0.5 --y-vel 0.0 --yaw-vel 0.0
```

`--load-checkpoint-path` accepts:

- Experiment root (`.../T1ForceJoystickFlatTerrain-...`)
- `checkpoints/` directory (auto-picks latest `000*` step folder)
- A specific step folder (`.../checkpoints/000202342400`)

Legacy `.pkl` models: `--load-model path/to/model.pkl` (deprecated).

### 3. Visualization (`--plot` + video)

```bash
python t1_joystick_flat_terrain_force.py --eval --plot \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000 \
  --x-vel 0.5 --y-vel 0.0 --yaw-vel 0.0
```

With `--plot`, the script writes figures and an MP4 into the **same experiment folder** as the checkpoint. Filename suffix reflects the command, e.g. `vx0.50_vy0.00_vyaw0.00`.

| Output file | Description |
|-------------|-------------|
| `force_data_vx*_vy*_vyaw*.csv` | Per-step forces, velocities, commands, contact stats |
| `force_analysis_vx*_....png` | Left/right Fz vs time |
| `force_step_averaged_vx*_....png` | Stance-normalized average Fz profiles |
| `impulse_trajectory_vx*_....png` | Impulse / penalty signal over full rollout |
| `impulse_zoom_one_step_vx*_....png` | Zoom on one contact pulse |
| `swing_peaks_vx*_....png` | Foot swing height per foot |
| `velocity_tracking_vx*_....png` | Command vs actual `dx, dy, dyaw` |
| `t1_force_policy_render_vx*_....mp4` | MuJoCo rollout video (contact points visible) |

### 4. Train + eval + plot in one command

If no mode flag is passed, the script runs **train, eval, and plot** together:

```bash
python t1_joystick_flat_terrain_force.py
```

### 5. Interactive command sweep

```bash
python t1_joystick_flat_terrain_force.py --interactive \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000
```

Runs several fixed `(vx, vy, vyaw)` settings and saves `interactive_force_commands.png`.

### 6. Policy “noise” metric

Measures how jittery foot forces are across episodes (lower = quieter contacts):

```bash
python t1_joystick_flat_terrain_force.py --noise-eval \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000 \
  --num-trajectories 10
```

Writes `noise_evaluation_summary.csv` in the experiment folder.

### 7. Compare two policies (CSV-only)

After running `--eval` on two checkpoints with the **same** velocity command:

```bash
python t1_joystick_flat_terrain_force.py \
  --compare-force-csv-a logs/run_a/force_data_vx0.50_vy0.00_vyaw0.00.csv \
  --compare-force-csv-b logs/run_b/force_data_vx0.50_vy0.00_vyaw0.00.csv \
  --compare-label-a "Fimp -0.1" \
  --compare-label-b "Fimp -0.3"
```

Produces `force_Fz_compare.png` and `velocity_compare.png`.

---

## Script reference (`learning/scripts/`)

### `t1_joystick_flat_terrain_force.py` (primary)

Full pipeline for **T1_Force** on flat terrain: PPO training, Brax checkpoint restore, force logging, plotting, video, comparisons, noise eval.

| Flag | Purpose |
|------|---------|
| `--train` | Run Brax PPO training |
| `--eval` | Roll out policy; save CSV |
| `--plot` | Generate all analysis figures (+ video if with `--eval`) |
| `--interactive` | Multi-command force plot |
| `--noise-eval` | Foot-force noise statistics |
| `--x-vel`, `--y-vel`, `--yaw-vel` | Joystick command for eval |
| `--load-checkpoint-path` | Brax checkpoint dir or experiment root |
| `--force-impulse-penalty-scale` | Override reward scale before train/eval |
| `--logdir` | Legacy TensorBoard dir name (default: `tensorboard_logs`; logs also go under `logs/`) |
| `--no-tensorboard` | Disable TensorBoard |
| `--compare-force-csv-a/b` | Two-policy CSV comparison |
| `--num-trajectories` | Episodes for `--noise-eval` |

**Main functions (internal):** `setup_training`, `train_policy`, `evaluate_policy`, `save_force_data_to_csv`, `plot_trajectory_analysis`, `plot_force_analysis`, `render_policy`, `evaluate_policy_noise`, `compare_force_csvs`, `interactive_commands`.

### `t1_joystick_flat_terrain_force_reward_only.py`

Thin wrapper around the force script that swaps the environment to **`T1ForceJoystickFlatTerrainForceRewardOnly`** (force affects **reward only**, not policy observations). Same CLI and outputs as above.

```bash
python t1_joystick_flat_terrain_force_reward_only.py --train
python t1_joystick_flat_terrain_force_reward_only.py --eval --plot \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrainForceRewardOnly-Fimp-0.3-steps200000000
```

### `t1_joystick_flat_terrain.py`

Baseline **T1 without force sensors** in the policy stack: train, eval, trajectory plots, video. Same `--train` / `--eval` / `--plot` / `--interactive` pattern; outputs under `logs/T1JoystickFlatTerrain-...`.

```bash
python t1_joystick_flat_terrain.py --train
python t1_joystick_flat_terrain.py --eval --plot --load-checkpoint-path logs/T1JoystickFlatTerrain-...
```

### `t1_joystick_flat_terrain_force_eval_zero_force.py`

Ablate force **observations** at test time: policy trained with force sensors, but evaluation **zeros** force inputs. Requires `--load-checkpoint-path`.

```bash
python t1_joystick_flat_terrain_force_eval_zero_force.py \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000 \
  --x-vel 0.5 --plot
```

---

## Generic Playground training (any env)

From `robot_feet/learning/`:

```bash
python train_jax_ppo.py --env_name T1ForceJoystickFlatTerrain
python train_jax_ppo.py --env_name CartpoleBalance
python train_jax_ppo.py --help
```

Checkpoints and logs go to `learning/logs/`. This is the upstream DeepMind entrypoint; the T1 scripts in `learning/scripts/` add force-specific logging, CSV export, and plotting.

**RSL-RL** (optional): `python train_rsl_rl.py --env_name=...` — see [`learning/README.md`](robot_feet/learning/README.md).

**Notebooks:** `robot_feet/learning/notebooks/` (locomotion, manipulation, vision).

**Sim2sim joysticks:** `robot_feet/mujoco_playground/experimental/sim2sim/play_t1_joystick.py` (CPU MuJoCo + gamepad).

---

## Isaac Gym path (`booster_gym/`)

Separate stack for **Isaac Gym** parallel training and real-robot deployment. Requires local `isaacgym/` install (see [`booster_gym/README.md`](booster_gym/README.md)).

| Stage | Command |
|-------|---------|
| Train | `python train.py --task=T1` |
| Play (Isaac) | `python play.py --task=T1 --checkpoint=-1` |
| Play (MuJoCo) | `python play_mujoco.py --task=T1 --checkpoint=-1` |
| Export for robot | `python export_model.py --task=T1 --checkpoint=-1` |
| Deploy | Follow [`booster_gym/deploy/README.md`](booster_gym/deploy/README.md) |

Logs: `booster_gym/logs/<timestamp>/`. Videos: `booster_gym/videos/`.

---

## Typical workflows

### A. Train a force-regularized policy and inspect feet

```bash
cd robot_feet/robot_feet/learning/scripts
python t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale -0.3
# after training finishes:
python t1_joystick_flat_terrain_force.py --eval --plot \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000 \
  --x-vel 0.5 --y-vel 0.0 --yaw-vel 0.0
tensorboard --logdir logs
```

### B. Reward-only forces (no force in observation)

```bash
python t1_joystick_flat_terrain_force_reward_only.py --train
python t1_joystick_flat_terrain_force_reward_only.py --eval --plot \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrainForceRewardOnly-...
```

### C. Baseline without force hardware in sim

```bash
python t1_joystick_flat_terrain.py --train
python t1_joystick_flat_terrain.py --eval --plot --load-checkpoint-path logs/T1JoystickFlatTerrain-...
```

### D. Does the policy need force observations at test time?

```bash
python t1_joystick_flat_terrain_force_eval_zero_force.py \
  --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-... --plot
```

---

## Dependencies (summary)

| Package | Used for |
|---------|----------|
| `jax` / `jaxlib` | GPU simulation + training |
| `mujoco`, `mujoco-mjx` | Physics |
| `brax` | PPO |
| `mediapy` | MP4 rendering |
| `matplotlib`, `pandas`, `numpy` | Plots and CSV |
| `tensorboardX` | Training curves |
| `orbax-checkpoint` | Brax checkpoints |

Full list: `robot_feet/pyproject.toml`, `learning/scripts/requirements.txt`.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| JAX reports `cpu` | Reinstall `jax[cuda12]`; check `nvidia-smi` |
| Unstable / non-reproducible training on RTX 30xx | `export JAX_DEFAULT_MATMUL_PRECISION=highest` |
| Empty or corrupt checkpoint | Retrain; use `--load-checkpoint-path` to a `000*` folder |
| Video render fails headless | `export MUJOCO_GL=egl` |
| Path not found for checkpoint | Paths are resolved relative to `learning/scripts/`; you can pass `logs/...` suffix only |

---

## Related documentation

- [`robot_feet/learning/scripts/README.md`](robot_feet/learning/scripts/README.md) — short scripts overview  
- [`robot_feet/learning/scripts/README_TensorBoard.md`](robot_feet/learning/scripts/README_TensorBoard.md) — TensorBoard metrics  
- [`robot_feet/learning/README.md`](robot_feet/learning/README.md) — Brax / RSL-RL trainers  
- [`robot_feet/README.md`](robot_feet/README.md) — upstream MuJoCo Playground install & citation  
- [`booster_gym/README.md`](booster_gym/README.md) — Isaac Gym train / deploy  

---

## Citation

If you use MuJoCo Playground in publications, cite the upstream project (see [`robot_feet/README.md`](robot_feet/README.md)). This repository extends Playground with Booster T1 force-sensor environments and experiment scripts described above.
