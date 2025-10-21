# TensorBoard Integration for T1 Robot Training

This document explains how to use TensorBoard to monitor T1 robot training progress.

## Features Added

The `t1_joystick_flat_terrain.py` script now includes comprehensive TensorBoard logging for:

### Training Metrics
- **Episode Reward**: Main performance metric
- **Episode Reward Std**: Standard deviation of rewards
- **Episode Length**: Average episode duration
- **Actor Loss**: Policy network loss
- **Critic Loss**: Value network loss
- **Entropy Loss**: Exploration bonus
- **Approx KL**: Policy change measurement
- **Clip Fraction**: PPO clipping frequency
- **Explained Variance**: Value function quality
- **Steps Per Second**: Training speed

### Usage

#### 1. Train with TensorBoard (Default)
```bash
python t1_joystick_flat_terrain.py --train
```

#### 2. Train with Custom Log Directory
```bash
python t1_joystick_flat_terrain.py --train --logdir my_experiment_logs
```

#### 3. Train without TensorBoard
```bash
python t1_joystick_flat_terrain.py --train --no-tensorboard
```

#### 4. View Training Progress
```bash
# In a separate terminal
tensorboard --logdir tensorboard_logs

# Then open http://localhost:6006 in your browser
```

## Log Directory Structure

```
tensorboard_logs/
└── t1_joystick_YYYYMMDD_HHMMSS/
    ├── events.out.tfevents.*
    └── ...
```

## Monitoring Training

### Key Metrics to Watch

1. **Training/Episode_Reward**: Should increase over time
2. **Training/Actor_Loss**: Should decrease (better policy)
3. **Training/Critic_Loss**: Should decrease (better value estimates)
4. **Training/Steps_Per_Second**: Training speed (higher is better)

### Troubleshooting

- **Flat reward curve**: Check if learning rate is too low
- **High actor loss**: May indicate unstable training
- **Low steps/second**: Consider reducing batch size or complexity

## TensorBoard Features

- **Scalars**: Plot training metrics over time
- **Histograms**: View parameter distributions
- **Images**: Visualize training data (if added)
- **Graphs**: Network architecture visualization

## Tips

1. **Multiple Experiments**: Use different `--logdir` values to compare runs
2. **Real-time Monitoring**: Keep TensorBoard open during training
3. **Hyperparameter Tuning**: Log different configurations to compare
4. **Early Stopping**: Monitor for convergence or overfitting

## Example Commands

```bash
# Full training pipeline with TensorBoard
python t1_joystick_flat_terrain.py --train --eval --plot

# Quick test run
python t1_joystick_flat_terrain.py --train --no-tensorboard

# Compare different configurations
python t1_joystick_flat_terrain.py --train --logdir experiment_1
python t1_joystick_flat_terrain.py --train --logdir experiment_2
tensorboard --logdir .  # View both experiments
```
