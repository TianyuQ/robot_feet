# Learning Scripts

This directory contains Python scripts for training and evaluating MuJoCo Playground environments.

## T1 Humanoid with Force Sensors

### `t1_joystick_flat_terrain.py`

A comprehensive script for training and evaluating the T1 humanoid robot with force sensors on flat terrain.

#### Features:
- **Force Sensor Integration**: Demonstrates 3D force measurement on both feet
- **Training Pipeline**: PPO training with domain randomization
- **Evaluation**: Policy rollout with force sensor data collection
- **Analysis**: Force sensor data visualization and statistics
- **Interactive Testing**: Test different velocity commands

#### Usage:

```bash
# Train the policy
python t1_joystick_flat_terrain.py --train

# Evaluate and plot results
python t1_joystick_flat_terrain.py --eval --plot

# Interactive command testing
python t1_joystick_flat_terrain.py --interactive

# Custom velocity commands
python t1_joystick_flat_terrain.py --eval --x-vel 1.0 --y-vel 0.5 --yaw-vel 2.0

# Run all (train, evaluate, and plot)
python t1_joystick_flat_terrain.py
```

#### Command Line Arguments:
- `--train`: Train the policy
- `--eval`: Evaluate the policy
- `--plot`: Plot force sensor analysis
- `--interactive`: Interactive command testing
- `--x-vel FLOAT`: X velocity command (default: 0.0)
- `--y-vel FLOAT`: Y velocity command (default: 0.0)
- `--yaw-vel FLOAT`: Yaw velocity command (default: 3.14)

#### Requirements:
- MuJoCo Playground
- JAX
- Brax
- Matplotlib
- NumPy

The script provides a complete demonstration of how force sensors enhance the T1 robot's locomotion capabilities through tactile feedback.
