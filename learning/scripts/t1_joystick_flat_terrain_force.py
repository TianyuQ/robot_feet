#!/usr/bin/env python3
"""
T1_Force Humanoid Locomotion with Force Sensors

This script demonstrates training and evaluation of the T1_Force humanoid robot with force sensors
on flat terrain. The T1_Force robot is equipped with force sensors on both feet to provide 
additional tactile feedback for improved balance and locomotion control.

Usage:
    python t1_joystick_flat_terrain_force.py [--train] [--eval] [--plot] [--interactive] [--logdir DIR] [--no-tensorboard]

Examples:
    # Train the policy with TensorBoard logging
    python t1_joystick_flat_terrain_force.py --train
    
    # Train with custom log directory
    python t1_joystick_flat_terrain_force.py --train --logdir my_logs
    
    # Train without TensorBoard
    python t1_joystick_flat_terrain_force.py --train --no-tensorboard
    
    # Evaluate and plot results
    python t1_joystick_flat_terrain_force.py --eval --plot
    
    # Interactive command testing
    python t1_joystick_flat_terrain_force.py --interactive
    
    # View TensorBoard (run in separate terminal)
    tensorboard --logdir tensorboard_logs
"""

import argparse
import functools
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add the project root to the Python path to use local mujoco_playground
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mujoco import mjx
import mujoco
from tensorboardX import SummaryWriter

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo


def print_section(title: str, width: int = 50) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


# Evaluation plots: fixed vertical-axis range for foot forces Fz (N)
EVAL_FORCE_FZ_YLIM = (-1000.0, 0.0)
# |mean(Fz)| below this (N) counts as ~no vertical load for pulse start/end detection
CONTACT_NEAR_ZERO_EPS_N = 100.0
# Stacked left/right Fz panels: wide for long rollouts, tall for readability
FORCE_FZ_STACK_FIGSIZE = (28.0, 12.0)

def resolve_load_checkpoint_path(user_path: str) -> str:
    """Resolve --load-checkpoint-path to a path that exists on disk.

    Tries, in order:
    1. Absolute path from cwd (``os.path.abspath``).
    2. Relative to this script's directory (``learning/scripts``).
    3. If the string contains ``logs/``, that suffix joined to the script directory
       (fixes duplicated ``robot_feet/learning/scripts/...`` when cwd is already
       ``learning/scripts``).
    """
    user_path = os.path.expanduser(user_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    norm = user_path.replace("\\", "/")
    candidates = [
        os.path.abspath(user_path),
        os.path.normpath(os.path.join(script_dir, user_path)),
    ]
    logs_idx = norm.find("logs/")
    if logs_idx != -1:
        candidates.append(os.path.normpath(os.path.join(script_dir, norm[logs_idx:])))
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    return os.path.normpath(candidates[0])


def load_trained_model(model_path: str) -> Tuple[callable, Dict, Dict]:
    """Load params and reconstruct make_inference_fn from config/env."""
    # Use cloudpickle to support serializing local/closure functions
    try:
        import cloudpickle as pickle  # type: ignore
    except Exception:
        import pickle  # fallback
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Guard against empty or truncated files
    try:
        size_bytes = os.path.getsize(model_path)
    except OSError:
        size_bytes = -1
    if size_bytes <= 0:
        raise RuntimeError(
            f"Model file exists but is empty or unreadable: {model_path}. "
            f"Please retrain to produce a valid checkpoint."
        )
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except EOFError as e:
        raise RuntimeError(
            f"Failed to load model (EOFError). The file may be truncated: {model_path}. "
            f"Size: {size_bytes} bytes. Retrain to regenerate the file."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to unpickle model file: {model_path}. The file may be corrupted or incompatible ({e})."
        ) from e
    
    print(f"Loaded trained model from: {model_path}")
    env_name = model_data.get('env_name', 'T1ForceJoystickFlatTerrain')
    metrics = model_data.get('metrics', {})
    params = model_data['params']
    saved_policy_vars = model_data.get('policy_vars')

    # Rebuild network factory from canonical config
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    # Build networks with env sizes to create an equivalent inference fn
    env = registry.load(env_name)
    networks = network_factory(env.observation_size, env.action_size)

    def _policy_variables_from_params(all_params):
        if isinstance(all_params, dict):
            if 'params' in all_params and isinstance(all_params['params'], dict):
                if 'policy' in all_params['params']:
                    return {'params': all_params['params']['policy']}
            if 'policy' in all_params:
                policy_tree = all_params['policy']
                if isinstance(policy_tree, dict) and 'params' in policy_tree:
                    return policy_tree
                return {'params': policy_tree}
        return {'params': all_params}

    def make_inference_fn_reconstructed(prms, deterministic=True):
        def policy(obs, rng):
            policy_vars = saved_policy_vars if saved_policy_vars is not None else _policy_variables_from_params(prms)
            act, _ = networks.policy_network.apply(policy_vars, obs)
            return act, None
        return policy

    print(f"Environment: {env_name}")
    print(f"Final reward: {metrics.get('eval/episode_reward', 'N/A')}")
    return make_inference_fn_reconstructed, params, metrics


def test_force_sensors(env) -> None:
    """Test and display force sensor readings."""
    print_section("Force Sensor Integration")
    
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Get force sensor readings
    left_force = env.get_left_foot_force(state.data)
    right_force = env.get_right_foot_force(state.data)
    feet_forces = env.get_feet_forces(state.data)

    print("Force Sensor Readings:")
    print("-" * 30)
    print(f"Left foot force: {left_force} (shape: {left_force.shape})")
    print(f"Right foot force: {right_force} (shape: {right_force.shape})")
    print(f"Combined feet forces: {feet_forces} (shape: {feet_forces.shape})")
    print(f"\nForce components: [Fx, Fy, Fz] (Newtons)")
    print(f"Left foot:  Fx={left_force[0]:.2f}, Fy={left_force[1]:.2f}, Fz={left_force[2]:.2f}")
    print(f"Right foot: Fx={right_force[0]:.2f}, Fy={right_force[1]:.2f}, Fz={right_force[2]:.2f}")


def setup_training(env_name: str, env_cfg, logdir: str = 'tensorboard_logs', enable_tensorboard: bool = True) -> Tuple[Dict, callable, SummaryWriter, str, str, str]:
    """Setup training configuration and function."""
    print_section("Training Configuration")
    
    # Get PPO training parameters
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    
    print("PPO Training Parameters:")
    print("-" * 30)
    for key, value in ppo_params.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    # Derive naming info for experiment folder
    # Total training steps (from PPO config)
    total_steps = ppo_params.get("num_timesteps", None)
    # Fz penalty scale (force_impulse_penalty) from env config; default to None if missing
    try:
        fz_penalty_scale = env_cfg.reward_config.scales.force_impulse_penalty
    except Exception:
        try:
            fz_penalty_scale = env_cfg["reward_config"]["scales"].get("force_impulse_penalty", None)
        except Exception:
            fz_penalty_scale = None

    # Build experiment name suffix with Fz penalty and total steps
    suffix_parts = []
    if fz_penalty_scale is not None:
        suffix_parts.append(f"Fimp{fz_penalty_scale:g}")
    if total_steps is not None:
        suffix_parts.append(f"steps{int(total_steps)}")
    suffix_str = "-".join(suffix_parts) if suffix_parts else "default"

    # Setup TensorBoard logging
    writer = None
    if enable_tensorboard:
        # Will be set after exp_root is created below
        writer = None
    else:
        print("\nTensorBoard logging disabled.")
    # Checkpoint directory (match train_jax_ppo style under logs), but include config info
    exp_name = f"{env_name}-{suffix_str}"
    exp_root = os.path.abspath(os.path.join("logs", exp_name))
    ckpt_path = os.path.join(exp_root, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")
    # Set TensorBoard logs to exp_root to unify outputs
    if enable_tensorboard:
        os.makedirs(exp_root, exist_ok=True)
        writer = SummaryWriter(exp_root)
        print(f"\nTensorBoard logging enabled. Logs saved to: {exp_root}")
        print(f"To view training progress, run: tensorboard --logdir logs")
    
    # Setup training progress tracking
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics.get("eval/episode_reward", 0.0))
        y_dataerr.append(metrics.get("eval/episode_reward_std", 0.0))

        # Log to TensorBoard
        if writer is not None:
            # Always log core metrics
            if "eval/episode_reward" in metrics:
                writer.add_scalar("Training/Episode_Reward", metrics["eval/episode_reward"], num_steps)
            if "eval/episode_reward_std" in metrics:
                writer.add_scalar("Training/Episode_Reward_Std", metrics["eval/episode_reward_std"], num_steps)
            if "eval/episode_length" in metrics:
                writer.add_scalar("Training/Episode_Length", metrics["eval/episode_length"], num_steps)
            
            # Log additional metrics if available
            if "eval/episode_reward_raw" in metrics:
                writer.add_scalar("Training/Episode_Reward_Raw", metrics["eval/episode_reward_raw"], num_steps)
            if "train/episode_reward" in metrics:
                writer.add_scalar("Training/Train_Episode_Reward", metrics["train/episode_reward"], num_steps)
            if "train/actor_loss" in metrics:
                writer.add_scalar("Training/Actor_Loss", metrics["train/actor_loss"], num_steps)
            if "train/critic_loss" in metrics:
                writer.add_scalar("Training/Critic_Loss", metrics["train/critic_loss"], num_steps)
            if "train/entropy_loss" in metrics:
                writer.add_scalar("Training/Entropy_Loss", metrics["train/entropy_loss"], num_steps)
            if "train/approx_kl" in metrics:
                writer.add_scalar("Training/Approx_KL", metrics["train/approx_kl"], num_steps)
            if "train/clipfrac" in metrics:
                writer.add_scalar("Training/Clip_Fraction", metrics["train/clipfrac"], num_steps)
            if "train/explained_variance" in metrics:
                writer.add_scalar("Training/Explained_Variance", metrics["train/explained_variance"], num_steps)
            
            # Log force sensor metrics if available
            if "eval/force_left_foot" in metrics:
                writer.add_scalar("Force/Left_Foot_Force", metrics["eval/force_left_foot"], num_steps)
            if "eval/force_right_foot" in metrics:
                writer.add_scalar("Force/Right_Foot_Force", metrics["eval/force_right_foot"], num_steps)
            if "eval/force_total" in metrics:
                writer.add_scalar("Force/Total_Force", metrics["eval/force_total"], num_steps)
            if "eval/force_contact_ratio" in metrics:
                writer.add_scalar("Force/Contact_Ratio", metrics["eval/force_contact_ratio"], num_steps)
            
            # Log training speed
            if len(times) > 1:
                time_diff = (times[-1] - times[-2]).total_seconds()
                steps_diff = x_data[-1] - x_data[-2] if len(x_data) > 1 else x_data[-1]
                steps_per_sec = steps_diff / time_diff if time_diff > 0 else 0
                writer.add_scalar("Training/Steps_Per_Second", steps_per_sec, num_steps)
            
            writer.flush()

        print(f"\rTraining step {num_steps}: Reward = {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}", end="")

    # Setup training function
    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
        save_checkpoint_path=ckpt_path,
    )
    
    return ppo_params, train_fn, writer, ckpt_path, exp_root


def train_policy(env, env_cfg, train_fn, writer=None, ckpt_path: str = None) -> Tuple[callable, Dict, Dict]:
    """Train the T1_Force policy with force sensors."""
    print_section("Training the Policy")
    
    print("Starting T1_Force training with force sensors...")
    
    start_time = time.time()
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load('T1ForceJoystickFlatTerrain', config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    end_time = time.time()
    
    # Log final training summary to TensorBoard
    if writer is not None:
        writer.add_scalar("Training/Final_Reward", metrics.get('eval/episode_reward', 0), 0)
        writer.add_scalar("Training/Total_Time", end_time - start_time, 0)
        writer.close()
    
    print(f"\nTraining completed!")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Final reward: {metrics.get('eval/episode_reward', 'N/A')}")
    if ckpt_path:
        print(f"Checkpoints saved under: {ckpt_path}")
    
    return make_inference_fn, params, metrics


def evaluate_policy(env, env_cfg, make_inference_fn, params, 
                   x_vel: float = 0.0, y_vel: float = 0.0, yaw_vel: float = 3.14) -> Tuple[List, List]:
    """Evaluate the trained policy and collect force sensor data."""
    print_section("Policy Evaluation")
    
    # Setup evaluation environment
    eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
    
    # JIT compile functions for faster execution
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    print("Evaluation environment setup complete.")
    
    def sample_pert(rng):
        rng, key1, key2 = jax.random.split(rng, 3)
        pert_mag = jax.random.uniform(
            key1, minval=0.0, maxval=0.0  # Disable velocity kick for now
        )
        duration_seconds = jax.random.uniform(
            key2, minval=0.05, maxval=0.2
        )
        duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
        state.info["pert_mag"] = pert_mag
        state.info["pert_duration"] = duration_steps
        state.info["pert_duration_seconds"] = duration_seconds
        return rng

    def has_perturbation_fields(st):
        return (
            isinstance(st.info, dict)
            and ("steps_since_last_pert" in st.info)
            and ("steps_until_next_pert" in st.info)
        )

    rng = jax.random.PRNGKey(0)
    rollout = []
    modify_scene_fns = []
    
    # Data collection lists
    swing_peak = []
    rewards = []
    linvel = []
    angvel = []
    track = []
    foot_vel = []
    rews = []
    contact = []
    feet_forces = []  # Force sensor data
    
    command = jp.array([x_vel, y_vel, yaw_vel])
    
    state = jit_reset(rng)
    if has_perturbation_fields(state):
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
    state.info["command"] = command
    
    print(f"Running evaluation with command: Vx={x_vel}, Vy={y_vel}, Vyaw={yaw_vel}")
    # Same rollout length as training (env_cfg.episode_length, e.g. 1000 steps)
    eval_num_steps = int(env_cfg.episode_length)
    print(
        f"Evaluation rollout: {eval_num_steps} steps (episode_length, dt={eval_env.dt})"
    )

    for i in range(eval_num_steps):
        if has_perturbation_fields(state):
            if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
                rng = sample_pert(rng)
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        
        # Collect data for analysis
        rews.append(
            {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
        )
        rollout.append(state)
        swing_peak.append(state.info["swing_peak"])
        rewards.append(
            {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
        )
        
        # Track force_impulse_penalty specifically
        if "reward/force_impulse_penalty" in state.metrics:
            # Store raw value for analysis
            pass  # Already in rewards dict
        linvel.append(eval_env.get_global_linvel(state.data))
        angvel.append(eval_env.get_gyro(state.data))
        track.append(
            eval_env._reward_tracking_lin_vel(
                state.info["command"], eval_env.get_local_linvel(state.data)
            )
        )
        
        # Foot velocity data
        feet_vel = state.data.sensordata[eval_env._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_vel.append(vel_norm)
        
        contact.append(state.info["last_contact"])
        
        # Force sensor data
        left_force = eval_env.get_left_foot_force(state.data)
        right_force = eval_env.get_right_foot_force(state.data)
        feet_forces.append(jp.hstack([left_force, right_force]))
        
        # Create modify scene functions for rendering
        xyz = np.array(state.data.xpos[eval_env._torso_body_id])
        xyz += np.array([0, 0, 0.2])
        x_axis = state.data.xmat[eval_env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=abs(state.info["command"][0]) / env_cfg.lin_vel_x[1] if env_cfg.lin_vel_x[1] > 0 else 0.1,
            )
        )
        
        if i % 100 == 0 or i == eval_num_steps - 1:
            print(f"\rStep {i + 1}/{eval_num_steps}", end="")
    
    print(f"\nEvaluation completed! Collected {len(rollout)} steps.")
    
    return rollout, {
        'swing_peak': swing_peak,
        'rewards': rewards,
        'linvel': linvel,
        'angvel': angvel,
        'track': track,
        'foot_vel': foot_vel,
        'rews': rews,
        'contact': contact,
        'feet_forces': feet_forces,
        'modify_scene_fns': modify_scene_fns
    }


def compute_episode_noise(force_data: jp.ndarray) -> float:
    """
    Compute noise metric from force data for a single episode.
    
    This function computes the noise/variability in force measurements.
    We use the standard deviation of force magnitudes as the noise metric,
    which captures how "noisy" or variable the forces are during the episode.
    
    Args:
        force_data: Array of shape (T, 6) where T is episode length and 6 is [Fx, Fy, Fz] for left and right feet.
    
    Returns:
        Episode noise metric (scalar float)
    """
    # Compute force magnitude for each foot at each timestep
    left_force_mag = jp.linalg.norm(force_data[:, :3], axis=1)  # (T,)
    right_force_mag = jp.linalg.norm(force_data[:, 3:], axis=1)  # (T,)
    
    # Combine both feet (sum of magnitudes)
    total_force_mag = left_force_mag + right_force_mag  # (T,)
    
    # Compute standard deviation as noise metric
    # Higher std dev = more noisy/variable forces
    episode_noise = jp.std(total_force_mag)
    
    return float(episode_noise)


def evaluate_policy_noise(env, env_cfg, make_inference_fn, params, 
                          num_trajectories: int = 10, 
                          seed: int = 42,
                          output_dir: str = ".") -> float:
    """
    Evaluate policy noise by sampling multiple control trajectories and computing average episode noise.
    
    This implements the noise evaluation pseudocode:
        Input: converged policy
        For controls in sampled_control_trajectories:
            force_data = rollout(controls)
            episode_noise = some_function(force_data)
        Compute average episode_noise
        Output: average episode_noise
    
    Args:
        env: Environment instance
        env_cfg: Environment configuration
        make_inference_fn: Function to create inference function from params
        params: Policy parameters
        num_trajectories: Number of control trajectories to sample and evaluate
        seed: Random seed for reproducibility
    
    Returns:
        Average episode noise across all sampled control trajectories
    """
    print_section("Noise Evaluation: Testing if Policy is 'Quieter'")
    print(f"Sampling {num_trajectories} control trajectories...")
    
    # Setup evaluation environment
    eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
    print(
        f"Each trajectory length: {env_cfg.episode_length} steps (episode_length, same as training, dt={eval_env.dt})"
    )
    
    # JIT compile functions for faster execution
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    # Get command ranges from environment config
    lin_vel_x_range = env_cfg.lin_vel_x if hasattr(env_cfg, 'lin_vel_x') else [-1.0, 1.0]
    lin_vel_y_range = env_cfg.lin_vel_y if hasattr(env_cfg, 'lin_vel_y') else [-1.0, 1.0]
    ang_vel_yaw_range = env_cfg.ang_vel_yaw if hasattr(env_cfg, 'ang_vel_yaw') else [-1.0, 1.0]
    
    def has_perturbation_fields(st):
        return (
            isinstance(st.info, dict)
            and ("steps_since_last_pert" in st.info)
            and ("steps_until_next_pert" in st.info)
        )
    
    # Sample control trajectories
    rng = jax.random.PRNGKey(seed)
    episode_noises = []
    
    for traj_idx in range(num_trajectories):
        # Sample a random control command
        rng, key1, key2, key3 = jax.random.split(rng, 4)
        x_vel = jax.random.uniform(key1, minval=lin_vel_x_range[0], maxval=lin_vel_x_range[1])
        y_vel = jax.random.uniform(key2, minval=lin_vel_y_range[0], maxval=lin_vel_y_range[1])
        yaw_vel = jax.random.uniform(key3, minval=ang_vel_yaw_range[0], maxval=ang_vel_yaw_range[1])
        
        # 10% chance to set command to zero (matching environment's sample_command behavior)
        rng, key4 = jax.random.split(rng)
        zero_command = jax.random.bernoulli(key4, p=0.1)
        command = jp.where(
            zero_command,
            jp.zeros(3),
            jp.array([x_vel, y_vel, yaw_vel])
        )
        
        # Run rollout with this control command
        traj_rng = jax.random.PRNGKey(seed + traj_idx)
        state = jit_reset(traj_rng)
        state.info["command"] = command
        
        # Collect force data during rollout (same episode_length as training / main eval)
        force_data = []

        for i in range(int(env_cfg.episode_length)):
            act_rng, traj_rng = jax.random.split(traj_rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            
            # Collect force sensor data
            left_force = eval_env.get_left_foot_force(state.data)
            right_force = eval_env.get_right_foot_force(state.data)
            force_data.append(jp.hstack([left_force, right_force]))
        
        # Compute episode noise from force data
        force_array = jp.array(force_data)  # Shape: (episode_length, 6)
        episode_noise = compute_episode_noise(force_array)
        episode_noises.append(episode_noise)
        
        # Save force data to CSV for this trajectory
        try:
            # Create a minimal eval_data dict for CSV export
            traj_eval_data = {
                'feet_forces': force_data,
                'linvel': [],
                'angvel': [],
                'rewards': [],
                'contact': []
            }
            # Create a minimal rollout list with command info
            class FakeState:
                def __init__(self, cmd):
                    self.info = {'command': cmd}
            traj_rollout = [FakeState(command)]
            csv_path = save_force_data_to_csv(
                traj_rollout, 
                traj_eval_data, 
                eval_env.dt, 
                output_dir
            )
            # Rename to include trajectory number
            new_csv_path = csv_path.replace('force_data_', f'force_data_traj{traj_idx+1:03d}_')
            if os.path.exists(csv_path):
                os.rename(csv_path, new_csv_path)
        except Exception as e:
            print(f"  Warning: Could not save CSV for trajectory {traj_idx + 1}: {e}")
        
        print(f"  Trajectory {traj_idx + 1}/{num_trajectories}: "
              f"command=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}), "
              f"episode_noise={episode_noise:.4f}")
    
    # Compute average episode noise
    average_episode_noise = float(jp.mean(jp.array(episode_noises)))
    std_episode_noise = float(jp.std(jp.array(episode_noises)))
    
    print("\n" + "=" * 50)
    print("Noise Evaluation Results:")
    print("-" * 50)
    print(f"Number of trajectories evaluated: {num_trajectories}")
    print(f"Average episode noise: {average_episode_noise:.4f}")
    print(f"Std dev of episode noise: {std_episode_noise:.4f}")
    print(f"Min episode noise: {float(jp.min(jp.array(episode_noises))):.4f}")
    print(f"Max episode noise: {float(jp.max(jp.array(episode_noises))):.4f}")
    print("=" * 50)
    print("\nInterpretation:")
    print("  Lower average episode noise indicates a 'quieter' policy")
    print("  (less variability in force measurements during locomotion)")
    
    # Save summary CSV with all episode noises
    try:
        summary_df = pd.DataFrame({
            'trajectory': range(1, num_trajectories + 1),
            'episode_noise': episode_noises
        })
        summary_csv_path = os.path.join(output_dir, "noise_evaluation_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nNoise evaluation summary saved to: {summary_csv_path}")
    except Exception as e:
        print(f"Warning: Could not save summary CSV: {e}")
    
    return average_episode_noise


def plot_force_analysis(force_data: List, dt: float, output_dir: str = ".", name_suffix: str = "") -> None:
    """Plot per-foot vertical force Fz only (no overlay comparison, no |F| panel)."""
    print_section("Force Sensor Analysis")
    
    force_data = jp.array(force_data)
    time_steps = jp.arange(len(force_data)) * dt

    fig, axes = plt.subplots(2, 1, figsize=FORCE_FZ_STACK_FIGSIZE, sharex=True)

    axes[0].plot(time_steps, force_data[:, 2], "b-", label="Fz")
    axes[0].set_title("Left foot Fz")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(EVAL_FORCE_FZ_YLIM)

    axes[1].plot(time_steps, force_data[:, 5], "b-", label="Fz")
    axes[1].set_title("Right foot Fz")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (N)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(EVAL_FORCE_FZ_YLIM)

    plt.tight_layout()
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    base = "force_analysis"
    fig_path = os.path.join(output_dir, f"{base}_{name_suffix}.png" if name_suffix else f"{base}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    left_dFz_dt = np.gradient(np.asarray(force_data[:, 2]), dt)
    right_dFz_dt = np.gradient(np.asarray(force_data[:, 5]), dt)

    print("Force Sensor Statistics (Fz only):")
    print("-" * 50)
    print("Left foot:")
    print(f"  Fz — mean: {jp.mean(force_data[:, 2]):.2f} N, min/max: {jp.min(force_data[:, 2]):.2f} / {jp.max(force_data[:, 2]):.2f} N")
    print(f"  |dFz/dt| — mean: {np.mean(np.abs(left_dFz_dt)):.2f} N/s, peak: {np.max(np.abs(left_dFz_dt)):.2f} N/s")
    print("Right foot:")
    print(f"  Fz — mean: {jp.mean(force_data[:, 5]):.2f} N, min/max: {jp.min(force_data[:, 5]):.2f} / {jp.max(force_data[:, 5]):.2f} N")
    print(f"  |dFz/dt| — mean: {np.mean(np.abs(right_dFz_dt)):.2f} N/s, peak: {np.max(np.abs(right_dFz_dt)):.2f} N/s")
    print(f"Fz sum (both feet) — mean: {jp.mean(force_data[:, 2] + force_data[:, 5]):.2f} N")
    print("-" * 50)
    plot_step_averaged_force_profiles(force_data, output_dir=output_dir, name_suffix=name_suffix)


def _extract_normalized_stance_profiles(
    fz: np.ndarray,
    contact_eps_n: float = CONTACT_NEAR_ZERO_EPS_N,
    n_points: int = 101,
    min_contact_samples: int = 4,
) -> np.ndarray:
    """
    Extract per-step stance profiles and normalize each step to 0-100% stance.
    Uses vertical load magnitude (-Fz), so higher values are larger compression force.
    """
    fz = np.asarray(fz, dtype=float)
    load = np.maximum(0.0, -fz)
    in_contact = load > float(contact_eps_n)

    starts: List[int] = []
    ends: List[int] = []
    for i in range(1, len(in_contact)):
        if (not in_contact[i - 1]) and in_contact[i]:
            starts.append(i)
        elif in_contact[i - 1] and (not in_contact[i]):
            ends.append(i)

    if in_contact[0]:
        starts = [0] + starts
    if in_contact[-1]:
        ends.append(len(in_contact))

    n_pairs = min(len(starts), len(ends))
    if n_pairs == 0:
        return np.empty((0, n_points), dtype=float)

    stance_profiles = []
    x_new = np.linspace(0.0, 1.0, n_points)
    for s, e in zip(starts[:n_pairs], ends[:n_pairs]):
        if e <= s:
            continue
        if (e - s) < min_contact_samples:
            continue
        seg = load[s:e]
        x_old = np.linspace(0.0, 1.0, len(seg))
        stance_profiles.append(np.interp(x_new, x_old, seg))

    if not stance_profiles:
        return np.empty((0, n_points), dtype=float)

    return np.vstack(stance_profiles)


def plot_step_averaged_force_profiles(
    force_data: np.ndarray,
    output_dir: str = ".",
    name_suffix: str = "",
) -> None:
    """
    Plot one-step vertical force analysis as mean ± SD across all detected steps
    in the episode, normalized to percentage of stance (similar to biomechanics papers).
    """
    F = np.asarray(force_data, dtype=float)
    left_profiles = _extract_normalized_stance_profiles(F[:, 2])
    right_profiles = _extract_normalized_stance_profiles(F[:, 5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    stance_pct = np.linspace(0.0, 100.0, 101)
    foot_info = [
        ("Left", left_profiles, "tab:blue"),
        ("Right", right_profiles, "tab:red"),
    ]

    max_y = 1.0
    for ax, (name, profiles, color) in zip(axes, foot_info):
        if profiles.shape[0] == 0:
            ax.text(0.5, 0.5, f"No valid {name.lower()} steps detected", ha="center", va="center")
            ax.set_title(f"{name} Foot (n=0)")
            ax.set_xlabel("Stance (%)")
            ax.grid(True, alpha=0.3)
            continue

        mean_curve = profiles.mean(axis=0)
        std_curve = profiles.std(axis=0)
        max_y = max(max_y, float(np.max(mean_curve + std_curve)))

        ax.plot(stance_pct, mean_curve, color=color, linewidth=2.0, label="Mean")
        ax.fill_between(
            stance_pct,
            np.maximum(0.0, mean_curve - std_curve),
            mean_curve + std_curve,
            color=color,
            alpha=0.25,
            label="Mean ± SD",
        )
        ax.set_title(f"{name} Foot (n={profiles.shape[0]} steps)")
        ax.set_xlabel("Stance (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, max_y * 1.05)
    axes[0].set_ylabel("Vertical load (-Fz) [N]")

    fig.suptitle("Step-Averaged Vertical Force Over Episode (Ensemble Mean ± SD)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = "force_step_averaged"
    fig_path = os.path.join(output_dir, f"{base}_{name_suffix}.png" if name_suffix else f"{base}.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print("Step-averaged force profile saved:")
    print(f"  {fig_path}")
    print(f"  Left steps: {left_profiles.shape[0]}, Right steps: {right_profiles.shape[0]}")


def _contact_pulse_window_indices(
    fz_l: np.ndarray,
    fz_r: np.ndarray,
    prefer_center: int,
    eps_n: float = CONTACT_NEAR_ZERO_EPS_N,
    pad_steps: int = 4,
    zoom_half_width_steps: int = 12,
) -> Tuple[int, int, str, Optional[Tuple[int, int]]]:
    """
    One “force pulse”: indices from leaving near-|F| (treated as ~0) to the next return to ~0,
    using |mean(Fz_L, Fz_R)|. Returns [i0, i1) slice indices with padding, a title note, and
    optional (pulse_start, pulse_end) sample indices for shading (end exclusive).
    """
    fz_l = np.asarray(fz_l, dtype=float)
    fz_r = np.asarray(fz_r, dtype=float)
    n = len(fz_l)
    m = np.abs(0.5 * (fz_l + fz_r))
    near = m < eps_n
    starts: List[int] = []
    ends: List[int] = []
    for i in range(1, n):
        if near[i - 1] and not near[i]:
            starts.append(i)
        if not near[i - 1] and near[i]:
            ends.append(i)

    pulses: List[Tuple[int, int]] = []
    for s in starts:
        after = [e for e in ends if e > s]
        if after:
            pulses.append((s, after[0]))

    if pulses:
        s0, e0 = min(pulses, key=lambda w: abs((w[0] + w[1]) / 2.0 - prefer_center))
        i_lo = max(0, s0 - pad_steps)
        i_hi = min(n, e0 + pad_steps)
        note = (
            f"|mean Fz| < {eps_n:g} N → load → back < {eps_n:g} N "
            f"(samples {s0}–{e0})"
        )
        return i_lo, i_hi, note, (s0, e0)

    # Fallback: no clear crossing — use mid-episode window
    k = max(1, min(prefer_center, n - 2))
    i_lo = max(0, k - zoom_half_width_steps)
    i_hi = min(n, k + zoom_half_width_steps + 1)
    return (
        i_lo,
        i_hi,
        f"no |mean Fz| < {eps_n:g} N crossing; using mid-rollout window",
        None,
    )


def plot_impulse_over_trajectory_and_zoom(
    eval_data: Dict,
    dt: float,
    output_dir: str = ".",
    name_suffix: str = "",
    zoom_half_width_steps: int = 12,
    force_impulse_scale: Optional[float] = None,
) -> None:
    """
    Evaluation impulse plots:

    1) **Physics:** vertical impulse per step \(F_z \\Delta t\) (N·s) per foot and mean.
    2) **Training cost:** raw = mean(Fz)/1000 as in `_cost_force_impulse_penalty` (no extra dt
       inside raw; global reward still multiplies the full sum by dt once).
    3) **This term's step reward:** raw × scale × dt.
    """
    feet = eval_data.get("feet_forces")
    if not feet or len(feet) < 2:
        return

    print_section("Impulse Over Trajectory (F·Δt and training cost)")

    scale = -0.1 if force_impulse_scale is None else float(force_impulse_scale)

    F = np.asarray(feet)
    fz_l = np.asarray(F[:, 2], dtype=float)
    fz_r = np.asarray(F[:, 5], dtype=float)
    n = len(fz_l)
    t = np.arange(n) * dt

    # Physical impulse (N·s) per step
    imp_l = fz_l * dt
    imp_r = fz_r * dt
    imp_mean = 0.5 * (fz_l + fz_r) * dt

    # Identical to env: jp.mean([fz_l, fz_r]) / 1e3
    raw_train = 0.5 * (fz_l + fz_r) / 1000.0
    step_contrib = raw_train * scale * dt

    rewards = eval_data.get("rewards", [])
    pen_raw = np.full(n, np.nan)
    for i, r in enumerate(rewards):
        if i >= n:
            break
        if isinstance(r, dict) and "force_impulse_penalty" in r:
            pen_raw[i] = float(r["force_impulse_penalty"])

    if np.any(np.isfinite(pen_raw)):
        delta = np.nanmax(np.abs(raw_train - pen_raw))
        if delta > 1e-4:
            print(
                f"  Note: max |recomputed raw − metrics raw| = {delta:.6f} "
                "(check sensor ordering if large)"
            )

    # Y-limit for impulse (N·s): match Fz cap [-1000, 0] → [-1000*dt, 0]
    fz_cap = abs(EVAL_FORCE_FZ_YLIM[0])
    imp_ylim = (-fz_cap * dt * 1.02, 0.0)

    fig1, axes1 = plt.subplots(3, 1, figsize=(28.0, 14.0), sharex=True)

    axes1[0].plot(t, imp_l, "b-", label="Left  Fz·Δt", linewidth=1.0)
    axes1[0].plot(t, imp_r, "r-", label="Right Fz·Δt", linewidth=1.0)
    axes1[0].plot(
        t,
        imp_mean,
        "k--",
        alpha=0.85,
        linewidth=1.2,
        label="mean(Fz)·Δt  (N·s)",
    )
    axes1[0].set_ylabel("Impulse (N·s)")
    axes1[0].set_title("Physical vertical impulse per simulation step (Fz × Δt)")
    axes1[0].legend(loc="best", fontsize=8)
    axes1[0].grid(True)
    axes1[0].set_ylim(imp_ylim)

    axes1[1].plot(
        t,
        raw_train,
        "b-",
        label="mean(Fz_L,Fz_R)/1000  (training _cost_force_impulse_penalty raw)",
        linewidth=1.2,
    )
    if np.any(np.isfinite(pen_raw)):
        axes1[1].plot(
            t,
            pen_raw,
            "g--",
            alpha=0.75,
            linewidth=1.0,
            label="force_impulse_penalty from metrics (should overlap)",
        )
    axes1[1].set_ylabel("Raw (before × scale)")
    axes1[1].set_title(
        "Training cost (instantaneous mean Fz / 1000; step reward applies one Δt to full sum)"
    )
    axes1[1].legend(loc="best", fontsize=7)
    axes1[1].grid(True)

    axes1[2].plot(
        t,
        step_contrib,
        "darkred",
        label=f"raw × scale × Δt   (scale={scale:g})",
        linewidth=1.2,
    )
    axes1[2].set_ylabel("This term in step reward")
    axes1[2].set_xlabel("Time (s)")
    axes1[2].legend(loc="best", fontsize=8)
    axes1[2].grid(True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base_traj = "impulse_trajectory"
    p1 = os.path.join(
        output_dir,
        f"{base_traj}_{name_suffix}.png" if name_suffix else f"{base_traj}.png",
    )
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Impulse trajectory figure saved to: {p1}")

    # Zoom: one contact pulse — from |mean Fz| ~ 0 through loading back to ~0 (see helper)
    i0, i1, zoom_note, pulse_ie = _contact_pulse_window_indices(
        fz_l,
        fz_r,
        prefer_center=n // 2,
        pad_steps=4,
        zoom_half_width_steps=zoom_half_width_steps,
    )
    t_win = t[i0:i1]

    fig2, (ax_fz, ax_imp, ax_raw) = plt.subplots(
        3,
        1,
        figsize=(28.0, 12.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]},
    )
    ax_fz.plot(t_win, fz_l[i0:i1], "b-", label="Left Fz", linewidth=1.2)
    ax_fz.plot(t_win, fz_r[i0:i1], "r-", label="Right Fz", linewidth=1.2)
    if pulse_ie is not None:
        ps, pe = pulse_ie
        ax_fz.axvspan(
            t[ps],
            t[min(pe, n - 1)],
            alpha=0.22,
            color="gray",
            label="pulse: low |F| → load → low |F|",
        )
        ax_fz.axvline(t[ps], color="k", linestyle=":", linewidth=0.9, alpha=0.65)
        ax_fz.axvline(t[min(pe, n - 1)], color="k", linestyle=":", linewidth=0.9, alpha=0.65)
    ax_fz.set_ylabel("Fz (N)")
    ax_fz.set_ylim(EVAL_FORCE_FZ_YLIM)
    ax_fz.set_title(f"Zoom: one force pulse ({zoom_note}); plot samples [{i0}, {i1})")
    ax_fz.legend(loc="upper right", fontsize=7)
    ax_fz.grid(True)

    ax_imp.plot(t_win, imp_l[i0:i1], "b-", label="Left Fz·Δt", linewidth=1.1)
    ax_imp.plot(t_win, imp_r[i0:i1], "r-", label="Right Fz·Δt", linewidth=1.1)
    ax_imp.plot(
        t_win,
        imp_mean[i0:i1],
        "k--",
        alpha=0.85,
        linewidth=1.1,
        label="mean(Fz)·Δt",
    )
    ax_imp.set_ylabel("Impulse (N·s)")
    ax_imp.set_ylim(imp_ylim)
    ax_imp.legend(loc="best", fontsize=8)
    ax_imp.grid(True)

    ax_raw.plot(t_win, raw_train[i0:i1], "b-", linewidth=1.2, label="mean(Fz)/1000")
    ax_raw.set_ylabel("Training raw")
    ax_raw.set_xlabel("Time (s)")
    ax_raw.legend(loc="best", fontsize=8)
    ax_raw.grid(True)

    plt.tight_layout()
    base_zoom = "impulse_zoom_one_step"
    p2 = os.path.join(
        output_dir,
        f"{base_zoom}_{name_suffix}.png" if name_suffix else f"{base_zoom}.png",
    )
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Force-pulse zoom figure saved to: {p2}")
    print(f"  {zoom_note}")


def analyze_force_impulse_penalty_contribution(rollout: List, eval_data: Dict, dt: float, force_impulse_scale: float = -0.1) -> None:
    """
    Analyze the contribution of force_impulse_penalty to the total reward.
    
    Args:
        rollout: List of rollout states (contains state.reward)
        eval_data: Dictionary containing evaluation data with rewards
        dt: Time step (seconds)
        force_impulse_scale: Scale factor for force_impulse_penalty (default: -0.1)
    """
    print_section("Force Impulse Penalty Reward Analysis")
    
    rewards = eval_data.get('rewards', [])
    if not rewards:
        print("No reward data available for analysis.")
        return
    
    # Extract force_impulse_penalty values (raw, before scaling)
    # Note: metrics store RAW values (before scaling), as seen in joystick.py line 400
    force_impulse_raw = []
    
    for reward_dict in rewards:
        if 'force_impulse_penalty' in reward_dict:
            force_impulse_raw.append(float(reward_dict['force_impulse_penalty']))
        else:
            force_impulse_raw.append(0.0)
    
    if not force_impulse_raw:
        print("force_impulse_penalty not found in reward data.")
        return
    
    force_impulse_raw = np.array(force_impulse_raw)
    
    # Get actual total rewards from rollout states
    # The reward computation in joystick.py:
    # 1. Raw values from _get_reward() 
    # 2. Scaled: rewards[k] = raw[k] * scales[k]  (line 369)
    # 3. Final: reward = sum(scaled_rewards) * dt  (line 371)
    # So state.reward is the final reward per step
    total_rewards_per_step = []
    for state in rollout:
        if hasattr(state, 'reward'):
            total_rewards_per_step.append(float(state.reward))
    
    if not total_rewards_per_step:
        print("Could not extract total rewards from rollout states.")
        return
    
    total_rewards_per_step = np.array(total_rewards_per_step)
    
    # Calculate scaled values and contribution
    # Scaled value = raw * scale
    force_impulse_scaled = force_impulse_raw * force_impulse_scale
    # Final contribution = scaled * dt
    force_impulse_contribution = force_impulse_scaled * dt
    
    # Calculate percentage contribution
    # Avoid division by zero
    nonzero_mask = np.abs(total_rewards_per_step) > 1e-10
    percentage_contribution = np.zeros_like(force_impulse_contribution)
    percentage_contribution[nonzero_mask] = (
        np.abs(force_impulse_contribution[nonzero_mask]) / 
        np.abs(total_rewards_per_step[nonzero_mask]) * 100
    )
    
    # Statistics
    print("Force Impulse Penalty Statistics:")
    print("-" * 60)
    print(f"Reward scale: {force_impulse_scale}")
    print(f"Time step (dt): {dt} s")
    print(f"\nRaw Values (before scaling):")
    print(f"  Mean: {np.mean(force_impulse_raw):.6f}")
    print(f"  Std:  {np.std(force_impulse_raw):.6f}")
    print(f"  Min:  {np.min(force_impulse_raw):.6f}")
    print(f"  Max:  {np.max(force_impulse_raw):.6f}")
    
    print(f"\nScaled Values (after scale={force_impulse_scale}):")
    print(f"  Mean: {np.mean(force_impulse_scaled):.6f}")
    print(f"  Std:  {np.std(force_impulse_scaled):.6f}")
    print(f"  Min:  {np.min(force_impulse_scaled):.6f}")
    print(f"  Max:  {np.max(force_impulse_scaled):.6f}")
    
    print(f"\nFinal Contribution (scaled * dt={dt}):")
    print(f"  Mean per step: {np.mean(force_impulse_contribution):.6f}")
    print(f"  Total over episode: {np.sum(force_impulse_contribution):.6f}")
    print(f"  Std: {np.std(force_impulse_contribution):.6f}")
    print(f"  Min: {np.min(force_impulse_contribution):.6f}")
    print(f"  Max: {np.max(force_impulse_contribution):.6f}")
    
    print(f"\nPercentage of Total Reward:")
    print(f"  Mean: {np.mean(percentage_contribution):.2f}%")
    print(f"  Std:  {np.std(percentage_contribution):.2f}%")
    print(f"  Min:  {np.min(percentage_contribution):.2f}%")
    print(f"  Max:  {np.max(percentage_contribution):.2f}%")
    
    # Compare with total reward
    total_episode_reward = np.sum(total_rewards_per_step)
    total_force_penalty = np.sum(force_impulse_contribution)
    overall_percentage = (np.abs(total_force_penalty) / np.abs(total_episode_reward) * 100) if np.abs(total_episode_reward) > 1e-10 else 0.0
    
    print(f"\nEpisode Summary:")
    print(f"  Total episode reward: {total_episode_reward:.4f}")
    print(f"  Total force_impulse_penalty contribution: {total_force_penalty:.4f}")
    print(f"  Overall percentage: {overall_percentage:.2f}%")
    print("-" * 60)
    
    # Interpretation
    print("\nInterpretation:")
    if overall_percentage < 1.0:
        print(f"  force_impulse_penalty contributes <1% to total reward (very small impact)")
    elif overall_percentage < 5.0:
        print(f"  force_impulse_penalty contributes ~{overall_percentage:.1f}% to total reward (small impact)")
    elif overall_percentage < 15.0:
        print(f"  force_impulse_penalty contributes ~{overall_percentage:.1f}% to total reward (moderate impact)")
    else:
        print(f"  force_impulse_penalty contributes ~{overall_percentage:.1f}% to total reward (significant impact)")
    
    print(f"  Current scale ({force_impulse_scale}) means the penalty is {'active' if force_impulse_scale != 0 else 'disabled'}")


def compare_force_csvs(csv_a: str, csv_b: str, label_a: str = "Policy A", label_b: str = "Policy B",
                       output_dir: Optional[str] = None) -> None:
    """
    Compare two policies using the rich data in their force CSVs.
    
    - Force plots: only Fz is shown (per foot).
    - Additional plots: velocity tracking vs command (dx, dy, dyaw) for both policies.
    """
    print_section("Policy Comparison From Force CSVs")

    if not os.path.exists(csv_a):
        print(f"First CSV not found: {csv_a}")
        return
    if not os.path.exists(csv_b):
        print(f"Second CSV not found: {csv_b}")
        return

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    
    # -----------------------------
    # 1) Force comparison (Fz only)
    # -----------------------------
    required_force_cols = ["time", "left_foot_Fz", "right_foot_Fz"]
    for col in required_force_cols:
        if col not in df_a.columns:
            print(f"Column '{col}' not found in first CSV: {csv_a}")
            return
        if col not in df_b.columns:
            print(f"Column '{col}' not found in second CSV: {csv_b}")
            return
    
    # Align by shortest length
    n = min(len(df_a), len(df_b))
    if n == 0:
        print("No data to compare (empty CSVs).")
        return
    
    t = df_a["time"].values[:n]
    
    left_fz_a = df_a["left_foot_Fz"].values[:n]
    left_fz_b = df_b["left_foot_Fz"].values[:n]
    right_fz_a = df_a["right_foot_Fz"].values[:n]
    right_fz_b = df_b["right_foot_Fz"].values[:n]
    
    fig_force, axes_force = plt.subplots(2, 1, figsize=FORCE_FZ_STACK_FIGSIZE, sharex=True)
    
    # Left foot Fz comparison
    axes_force[0].plot(t, left_fz_a, "b-", label=f"{label_a} Left Fz")
    axes_force[0].plot(t, left_fz_b, "r--", label=f"{label_b} Left Fz")
    axes_force[0].set_title("Left Foot Fz Comparison")
    axes_force[0].set_ylabel("Fz (N)")
    axes_force[0].legend()
    axes_force[0].grid(True)
    
    # Right foot Fz comparison
    axes_force[1].plot(t, right_fz_a, "b-", label=f"{label_a} Right Fz")
    axes_force[1].plot(t, right_fz_b, "r--", label=f"{label_b} Right Fz")
    axes_force[1].set_title("Right Foot Fz Comparison")
    axes_force[1].set_xlabel("Time (s)")
    axes_force[1].set_ylabel("Fz (N)")
    axes_force[1].legend()
    axes_force[1].grid(True)
    for ax in axes_force:
        ax.set_ylim(EVAL_FORCE_FZ_YLIM)
    
    plt.tight_layout()
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_a))
    os.makedirs(output_dir, exist_ok=True)
    
    force_fig_path = os.path.join(output_dir, "force_Fz_compare.png")
    fig_force.savefig(force_fig_path, dpi=150)
    plt.close(fig_force)
    
    print(f"Force comparison figure saved to: {force_fig_path}")
    
    # -----------------------------------
    # 2) Kinematics / command comparison
    # -----------------------------------
    vel_cols = ["linvel_x", "linvel_y", "angvel_z", "command_x", "command_y", "command_yaw"]
    if all(col in df_a.columns for col in vel_cols) and all(col in df_b.columns for col in vel_cols):
        linvel_x_a = df_a["linvel_x"].values[:n]
        linvel_y_a = df_a["linvel_y"].values[:n]
        angvel_z_a = df_a["angvel_z"].values[:n]
        
        linvel_x_b = df_b["linvel_x"].values[:n]
        linvel_y_b = df_b["linvel_y"].values[:n]
        angvel_z_b = df_b["angvel_z"].values[:n]
        
        # Commands (typically constant over rollout); read from A
        cmd_x = df_a["command_x"].values[:n]
        cmd_y = df_a["command_y"].values[:n]
        cmd_yaw = df_a["command_yaw"].values[:n]
        
        fig_vel, axes_vel = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # dx
        axes_vel[0].plot(t, linvel_x_a, "b-", label=f"{label_a} v_x")
        axes_vel[0].plot(t, linvel_x_b, "r--", label=f"{label_b} v_x")
        axes_vel[0].plot(t, cmd_x, "k:", label="command x")
        axes_vel[0].set_ylabel("v_x (m/s)")
        axes_vel[0].set_title("Forward Velocity Comparison")
        axes_vel[0].legend()
        axes_vel[0].grid(True)
        
        # dy
        axes_vel[1].plot(t, linvel_y_a, "b-", label=f"{label_a} v_y")
        axes_vel[1].plot(t, linvel_y_b, "r--", label=f"{label_b} v_y")
        axes_vel[1].plot(t, cmd_y, "k:", label="command y")
        axes_vel[1].set_ylabel("v_y (m/s)")
        axes_vel[1].set_title("Lateral Velocity Comparison")
        axes_vel[1].legend()
        axes_vel[1].grid(True)
        
        # dyaw
        axes_vel[2].plot(t, angvel_z_a, "b-", label=f"{label_a} yaw rate")
        axes_vel[2].plot(t, angvel_z_b, "r--", label=f"{label_b} yaw rate")
        axes_vel[2].plot(t, cmd_yaw, "k:", label="command yaw")
        axes_vel[2].set_ylabel("yaw rate (rad/s)")
        axes_vel[2].set_xlabel("Time (s)")
        axes_vel[2].set_title("Yaw Velocity Comparison")
        axes_vel[2].legend()
        axes_vel[2].grid(True)
        
        plt.tight_layout()
        
        vel_fig_path = os.path.join(output_dir, "velocity_compare.png")
        fig_vel.savefig(vel_fig_path, dpi=150)
        plt.close(fig_vel)
        
        print(f"Velocity / command comparison figure saved to: {vel_fig_path}")


def save_force_data_to_csv(rollout: List, eval_data: Dict, dt: float, output_dir: str = ".", name_suffix: str = "") -> str:
    """
    Save force sensor data and other evaluation metrics to CSV file.
    
    Args:
        rollout: List of rollout states
        eval_data: Dictionary containing evaluation data (forces, velocities, etc.)
        dt: Time step (seconds)
        output_dir: Directory to save CSV file
    
    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert force data to numpy array
    force_data = jp.array(eval_data.get('feet_forces', []))
    num_steps = len(force_data)
    
    # Create time array
    time_steps = np.arange(num_steps) * dt
    
    # Extract other data
    linvel = np.array(eval_data.get('linvel', []))
    angvel = np.array(eval_data.get('angvel', []))
    rewards = eval_data.get('rewards', [])
    contact = eval_data.get('contact', [])
    
    # Extract command from rollout if available
    command = None
    if rollout and hasattr(rollout[0], 'info') and 'command' in rollout[0].info:
        command = rollout[0].info['command']
    
    # Create DataFrame
    data_dict = {
        'time': time_steps,
        'left_foot_Fx': force_data[:, 0] if num_steps > 0 else [],
        'left_foot_Fy': force_data[:, 1] if num_steps > 0 else [],
        'left_foot_Fz': force_data[:, 2] if num_steps > 0 else [],
        'right_foot_Fx': force_data[:, 3] if num_steps > 0 else [],
        'right_foot_Fy': force_data[:, 4] if num_steps > 0 else [],
        'right_foot_Fz': force_data[:, 5] if num_steps > 0 else [],
    }
    
    # Add force magnitudes
    if num_steps > 0:
        left_mag = jp.sqrt(jp.sum(force_data[:, :3]**2, axis=1))
        right_mag = jp.sqrt(jp.sum(force_data[:, 3:]**2, axis=1))
        data_dict['left_foot_magnitude'] = left_mag
        data_dict['right_foot_magnitude'] = right_mag
        data_dict['total_force_magnitude'] = left_mag + right_mag
        
        # Calculate dF/dt (force rate of change) for each component
        # Using numpy gradient for better accuracy
        left_dFx_dt = np.gradient(force_data[:, 0], dt)
        left_dFy_dt = np.gradient(force_data[:, 1], dt)
        left_dFz_dt = np.gradient(force_data[:, 2], dt)
        right_dFx_dt = np.gradient(force_data[:, 3], dt)
        right_dFy_dt = np.gradient(force_data[:, 4], dt)
        right_dFz_dt = np.gradient(force_data[:, 5], dt)
        
        data_dict['left_foot_dFx_dt'] = left_dFx_dt
        data_dict['left_foot_dFy_dt'] = left_dFy_dt
        data_dict['left_foot_dFz_dt'] = left_dFz_dt
        data_dict['right_foot_dFx_dt'] = right_dFx_dt
        data_dict['right_foot_dFy_dt'] = right_dFy_dt
        data_dict['right_foot_dFz_dt'] = right_dFz_dt
        
        # dF/dt for force magnitudes
        data_dict['left_foot_dmagnitude_dt'] = np.gradient(left_mag, dt)
        data_dict['right_foot_dmagnitude_dt'] = np.gradient(right_mag, dt)
        data_dict['total_dmagnitude_dt'] = np.gradient(left_mag + right_mag, dt)
        
        # Calculate average and peak forces (for each timestep, store running stats)
        # Average force up to current time (cumulative mean)
        left_avg_Fx = np.cumsum(force_data[:, 0]) / np.arange(1, num_steps + 1)
        left_avg_Fy = np.cumsum(force_data[:, 1]) / np.arange(1, num_steps + 1)
        left_avg_Fz = np.cumsum(force_data[:, 2]) / np.arange(1, num_steps + 1)
        right_avg_Fx = np.cumsum(force_data[:, 3]) / np.arange(1, num_steps + 1)
        right_avg_Fy = np.cumsum(force_data[:, 4]) / np.arange(1, num_steps + 1)
        right_avg_Fz = np.cumsum(force_data[:, 5]) / np.arange(1, num_steps + 1)
        left_avg_mag = np.cumsum(left_mag) / np.arange(1, num_steps + 1)
        right_avg_mag = np.cumsum(right_mag) / np.arange(1, num_steps + 1)
        total_avg_mag = np.cumsum(left_mag + right_mag) / np.arange(1, num_steps + 1)
        
        data_dict['left_foot_avg_Fx'] = left_avg_Fx
        data_dict['left_foot_avg_Fy'] = left_avg_Fy
        data_dict['left_foot_avg_Fz'] = left_avg_Fz
        data_dict['right_foot_avg_Fx'] = right_avg_Fx
        data_dict['right_foot_avg_Fy'] = right_avg_Fy
        data_dict['right_foot_avg_Fz'] = right_avg_Fz
        data_dict['left_foot_avg_magnitude'] = left_avg_mag
        data_dict['right_foot_avg_magnitude'] = right_avg_mag
        data_dict['total_avg_magnitude'] = total_avg_mag
        
        # Peak force up to current time (running maximum)
        left_peak_Fx = np.maximum.accumulate(np.abs(force_data[:, 0]))
        left_peak_Fy = np.maximum.accumulate(np.abs(force_data[:, 1]))
        left_peak_Fz = np.maximum.accumulate(np.abs(force_data[:, 2]))
        right_peak_Fx = np.maximum.accumulate(np.abs(force_data[:, 3]))
        right_peak_Fy = np.maximum.accumulate(np.abs(force_data[:, 4]))
        right_peak_Fz = np.maximum.accumulate(np.abs(force_data[:, 5]))
        left_peak_mag = np.maximum.accumulate(left_mag)
        right_peak_mag = np.maximum.accumulate(right_mag)
        total_peak_mag = np.maximum.accumulate(left_mag + right_mag)
        
        data_dict['left_foot_peak_Fx'] = left_peak_Fx
        data_dict['left_foot_peak_Fy'] = left_peak_Fy
        data_dict['left_foot_peak_Fz'] = left_peak_Fz
        data_dict['right_foot_peak_Fx'] = right_peak_Fx
        data_dict['right_foot_peak_Fy'] = right_peak_Fy
        data_dict['right_foot_peak_Fz'] = right_peak_Fz
        data_dict['left_foot_peak_magnitude'] = left_peak_mag
        data_dict['right_foot_peak_magnitude'] = right_peak_mag
        data_dict['total_peak_magnitude'] = total_peak_mag
    else:
        data_dict['left_foot_magnitude'] = []
        data_dict['right_foot_magnitude'] = []
        data_dict['total_force_magnitude'] = []
        data_dict['left_foot_dFx_dt'] = []
        data_dict['left_foot_dFy_dt'] = []
        data_dict['left_foot_dFz_dt'] = []
        data_dict['right_foot_dFx_dt'] = []
        data_dict['right_foot_dFy_dt'] = []
        data_dict['right_foot_dFz_dt'] = []
        data_dict['left_foot_dmagnitude_dt'] = []
        data_dict['right_foot_dmagnitude_dt'] = []
        data_dict['total_dmagnitude_dt'] = []
        data_dict['left_foot_avg_Fx'] = []
        data_dict['left_foot_avg_Fy'] = []
        data_dict['left_foot_avg_Fz'] = []
        data_dict['right_foot_avg_Fx'] = []
        data_dict['right_foot_avg_Fy'] = []
        data_dict['right_foot_avg_Fz'] = []
        data_dict['left_foot_avg_magnitude'] = []
        data_dict['right_foot_avg_magnitude'] = []
        data_dict['total_avg_magnitude'] = []
        data_dict['left_foot_peak_Fx'] = []
        data_dict['left_foot_peak_Fy'] = []
        data_dict['left_foot_peak_Fz'] = []
        data_dict['right_foot_peak_Fx'] = []
        data_dict['right_foot_peak_Fy'] = []
        data_dict['right_foot_peak_Fz'] = []
        data_dict['left_foot_peak_magnitude'] = []
        data_dict['right_foot_peak_magnitude'] = []
        data_dict['total_peak_magnitude'] = []
    
    # Add velocities if available
    if len(linvel) > 0:
        data_dict['linvel_x'] = linvel[:, 0] if linvel.shape[1] > 0 else np.zeros(num_steps)
        data_dict['linvel_y'] = linvel[:, 1] if linvel.shape[1] > 1 else np.zeros(num_steps)
        data_dict['linvel_z'] = linvel[:, 2] if linvel.shape[1] > 2 else np.zeros(num_steps)
    else:
        data_dict['linvel_x'] = np.zeros(num_steps)
        data_dict['linvel_y'] = np.zeros(num_steps)
        data_dict['linvel_z'] = np.zeros(num_steps)
    
    if len(angvel) > 0:
        data_dict['angvel_x'] = angvel[:, 0] if angvel.shape[1] > 0 else np.zeros(num_steps)
        data_dict['angvel_y'] = angvel[:, 1] if angvel.shape[1] > 1 else np.zeros(num_steps)
        data_dict['angvel_z'] = angvel[:, 2] if angvel.shape[1] > 2 else np.zeros(num_steps)
    else:
        data_dict['angvel_x'] = np.zeros(num_steps)
        data_dict['angvel_y'] = np.zeros(num_steps)
        data_dict['angvel_z'] = np.zeros(num_steps)
    
    # Add command if available
    if command is not None:
        data_dict['command_x'] = np.full(num_steps, command[0])
        data_dict['command_y'] = np.full(num_steps, command[1]) if len(command) > 1 else np.zeros(num_steps)
        data_dict['command_yaw'] = np.full(num_steps, command[2]) if len(command) > 2 else np.zeros(num_steps)
    else:
        data_dict['command_x'] = np.zeros(num_steps)
        data_dict['command_y'] = np.zeros(num_steps)
        data_dict['command_yaw'] = np.zeros(num_steps)
    
    # Add contact information if available
    if len(contact) > 0:
        # Contact is typically a list of arrays, convert to single values
        if isinstance(contact[0], (list, np.ndarray, jp.ndarray)):
            # Take first contact value or sum if multiple
            contact_vals = [float(c[0]) if len(c) > 0 else 0.0 for c in contact]
        else:
            contact_vals = [float(c) for c in contact]
        data_dict['contact'] = contact_vals[:num_steps] if len(contact_vals) >= num_steps else contact_vals + [0.0] * (num_steps - len(contact_vals))
    else:
        data_dict['contact'] = np.zeros(num_steps)
    
    # Add force_impulse_penalty reward data if available
    rewards = eval_data.get('rewards', [])
    if rewards and len(rewards) > 0:
        force_impulse_raw = []
        force_impulse_scaled = []
        force_impulse_contribution = []
        total_reward_per_step = []
        
        for i, reward_dict in enumerate(rewards):
            # Raw value (before scaling)
            raw_val = float(reward_dict.get('force_impulse_penalty', 0.0))
            force_impulse_raw.append(raw_val)
            
            # Scaled value (raw * scale, where scale = -0.1)
            scaled_val = raw_val * -0.1
            force_impulse_scaled.append(scaled_val)
            
            # Final contribution (scaled * dt)
            contribution = scaled_val * dt
            force_impulse_contribution.append(contribution)
            
            # Total reward from state if available
            if i < len(rollout) and hasattr(rollout[i], 'reward'):
                total_reward_per_step.append(float(rollout[i].reward))
            else:
                total_reward_per_step.append(0.0)
        
        # Pad or truncate to match num_steps
        if len(force_impulse_raw) < num_steps:
            force_impulse_raw.extend([0.0] * (num_steps - len(force_impulse_raw)))
            force_impulse_scaled.extend([0.0] * (num_steps - len(force_impulse_scaled)))
            force_impulse_contribution.extend([0.0] * (num_steps - len(force_impulse_contribution)))
            total_reward_per_step.extend([0.0] * (num_steps - len(total_reward_per_step)))
        elif len(force_impulse_raw) > num_steps:
            force_impulse_raw = force_impulse_raw[:num_steps]
            force_impulse_scaled = force_impulse_scaled[:num_steps]
            force_impulse_contribution = force_impulse_contribution[:num_steps]
            total_reward_per_step = total_reward_per_step[:num_steps]
        
        data_dict['force_impulse_penalty_raw'] = np.array(force_impulse_raw)
        data_dict['force_impulse_penalty_scaled'] = np.array(force_impulse_scaled)
        data_dict['force_impulse_penalty_contribution'] = np.array(force_impulse_contribution)
        data_dict['total_reward_per_step'] = np.array(total_reward_per_step)
        
        # Calculate percentage contribution
        percentage_contrib = np.zeros(num_steps)
        for i in range(num_steps):
            if abs(total_reward_per_step[i]) > 1e-10:
                percentage_contrib[i] = abs(force_impulse_contribution[i]) / abs(total_reward_per_step[i]) * 100
        data_dict['force_impulse_penalty_percentage'] = percentage_contrib
    else:
        data_dict['force_impulse_penalty_raw'] = np.zeros(num_steps)
        data_dict['force_impulse_penalty_scaled'] = np.zeros(num_steps)
        data_dict['force_impulse_penalty_contribution'] = np.zeros(num_steps)
        data_dict['total_reward_per_step'] = np.zeros(num_steps)
        data_dict['force_impulse_penalty_percentage'] = np.zeros(num_steps)
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Save to CSV
    base = "force_data"
    csv_path = os.path.join(output_dir, f"{base}_{name_suffix}.csv" if name_suffix else f"{base}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nForce data saved to CSV: {csv_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Includes: forces (Fx, Fy, Fz), magnitudes, dF/dt, average forces, peak forces")
    print(f"  Plus: velocities, commands, contact data")
    
    return csv_path


def plot_trajectory_analysis(
    rollout: List,
    eval_data: Dict,
    dt: float,
    output_dir: str = ".",
    name_suffix: str = "",
    force_impulse_scale: Optional[float] = None,
) -> None:
    """Plot trajectory analysis for the robot following the locomotion notebook pattern."""
    print_section("Trajectory Analysis")
    
    swing_peak = jp.array(eval_data['swing_peak'])
    linvel = jp.array(eval_data['linvel'])
    angvel = jp.array(eval_data['angvel'])
    
    # Plot each foot in a 2x2 grid (for humanoid, we'll adapt this)
    if len(swing_peak[0]) >= 2:  # Check if we have foot data
        names = ["Left", "Right"] if len(swing_peak[0]) == 2 else ["FR", "FL", "RR", "RL"]
        colors = ["r", "g", "b", "y"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axs.flat):
            if i < len(swing_peak[0]):
                ax.plot(swing_peak[:, i], color=colors[i])
                ax.set_ylim([0, 0.15])  # Adjust based on expected foot height
                ax.set_title(names[i])
                ax.set_xlabel("time")
                ax.set_ylabel("height")
            else:
                ax.set_visible(False)
        plt.tight_layout()
        # Save swing peak figure
        os.makedirs(output_dir, exist_ok=True)
        base = "swing_peaks"
        fig_path = os.path.join(output_dir, f"{base}_{name_suffix}.png" if name_suffix else f"{base}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    linvel_x = linvel[:, 0]
    linvel_y = linvel[:, 1]
    angvel_yaw = angvel[:, 2]

    # Plot whether velocity is within the command range
    linvel_x = jp.convolve(linvel_x, jp.ones(10) / 10, mode="same")
    linvel_y = jp.convolve(linvel_y, jp.ones(10) / 10, mode="same")
    angvel_yaw = jp.convolve(angvel_yaw, jp.ones(10) / 10, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(linvel_x)
    axes[1].plot(linvel_y)
    axes[2].plot(angvel_yaw)

    # Set y-axis limits based on command ranges
    axes[0].set_ylim(-1.0, 1.0)
    axes[1].set_ylim(-0.8, 0.8)
    axes[2].set_ylim(-1.0, 1.0)

    # Add command lines
    if rollout:
        command = rollout[0].info["command"]
        axes[0].axhline(command[0], color="red", linestyle="--")
        axes[1].axhline(command[1], color="red", linestyle="--")
        axes[2].axhline(command[2], color="red", linestyle="--")

    labels = ["dx", "dy", "dyaw"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])
    
    plt.tight_layout()
    # Save velocity tracking figure
    os.makedirs(output_dir, exist_ok=True)
    base = "velocity_tracking"
    fig_path2 = os.path.join(output_dir, f"{base}_{name_suffix}.png" if name_suffix else f"{base}.png")
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)

    # Impulse: same raw/scaled signal as training force_impulse_penalty
    if "feet_forces" in eval_data and eval_data["feet_forces"]:
        plot_impulse_over_trajectory_and_zoom(
            eval_data,
            dt,
            output_dir,
            name_suffix=name_suffix,
            force_impulse_scale=force_impulse_scale,
        )

    # Force sensor analysis
    if 'feet_forces' in eval_data and eval_data['feet_forces']:
        plot_force_analysis(eval_data['feet_forces'], dt, output_dir, name_suffix=name_suffix)

    # Print trajectory statistics
    print("Trajectory Statistics:")
    print("-" * 30)
    if rollout:
        positions = np.array([state.data.qpos[:3] for state in rollout])
        print(f"Total distance traveled: {np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)):.2f} m")
        print(f"Final position: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f})")
        
        rewards = np.array([state.reward for state in rollout])
        print(f"Average reward: {np.mean(rewards):.3f}")
        print(f"Total reward: {np.sum(rewards):.3f}")


def render_policy(rollout: List, eval_data: Dict, eval_env, output_dir: str = ".", name_suffix: str = "") -> None:
    """Render the policy following the locomotion notebook pattern."""
    print_section("Policy Rendering")
    
    modify_scene_fns = eval_data.get('modify_scene_fns', [])
    
    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every] if modify_scene_fns else []
    
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    
    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
        modify_scene_fns=mod_fns,
    )
    
    # Save video instead of showing it (since we're in a script, not notebook)
    import mediapy as media
    os.makedirs(output_dir, exist_ok=True)
    base = "t1_force_policy_render"
    output_path = os.path.join(output_dir, f"{base}_{name_suffix}.mp4" if name_suffix else f"{base}.mp4")
    media.write_video(output_path, frames, fps=fps)
    print(f"Video saved to: {output_path}")


def interactive_commands(env, env_cfg, make_inference_fn, params) -> None:
    """Test the robot with different velocity commands."""
    print_section("Interactive Command Testing")
    
    def test_commands(x_vel, y_vel, yaw_vel, duration=5.0):
        """Test robot with specific velocity commands."""
        eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        
        rng = jax.random.PRNGKey(0)
        rollout = []
        force_data = []
        
        command = jp.array([x_vel, y_vel, yaw_vel])
        state = jit_reset(rng)
        state.info["command"] = command
        
        steps = int(duration / eval_env.dt)
        
        for i in range(steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            
            rollout.append(state)
            force_data.append(eval_env.get_feet_forces(state.data))
        
        return rollout, jp.array(force_data)

    # Test different commands
    commands = [
        (0.0, 0.0, 0.0),    # Stand still
        (1.0, 0.0, 0.0),    # Forward
        (0.0, 1.0, 0.0),    # Sideways
        (0.0, 0.0, 1.0),    # Turn
        (1.0, 0.0, 1.0),    # Forward + Turn
    ]
    
    fig, axes = plt.subplots(len(commands), 1, figsize=(12, 3*len(commands)))
    if len(commands) == 1:
        axes = [axes]

    for i, (x_vel, y_vel, yaw_vel) in enumerate(commands):
        print(f"Testing command: Vx={x_vel}, Vy={y_vel}, Vyaw={yaw_vel}")
        rollout, forces = test_commands(x_vel, y_vel, yaw_vel, duration=3.0)
        time_steps = jp.arange(len(forces)) * env.dt
        
        axes[i].plot(time_steps, forces[:, 2], 'r-', label='Left Fz')
        axes[i].plot(time_steps, forces[:, 5], 'b-', label='Right Fz')
        axes[i].set_title(f'Command: Vx={x_vel}, Vy={y_vel}, Vyaw={yaw_vel}')
        axes[i].set_ylabel('Vertical Force (N)')
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Time (s)')
    for ax in axes:
        ax.set_ylim(EVAL_FORCE_FZ_YLIM)
    plt.tight_layout()
    _interactive_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(_interactive_out, exist_ok=True)
    _interactive_png = os.path.join(_interactive_out, "interactive_force_commands.png")
    fig.savefig(_interactive_png, dpi=150)
    plt.close(fig)
    print(f"Interactive force plot saved to: {_interactive_png}")

    print("Force sensor data shows how the robot adapts its foot forces")
    print("based on different locomotion commands.")


def main():
    """Main function to run T1_Force joystick with force sensors."""
    parser = argparse.ArgumentParser(description='T1_Force Humanoid Locomotion with Force Sensors')
    parser.add_argument('--train', action='store_true', help='Train the policy')
    parser.add_argument('--eval', action='store_true', help='Evaluate the policy')
    parser.add_argument('--plot', action='store_true', help='Plot force sensor analysis')
    parser.add_argument('--interactive', action='store_true', help='Interactive command testing')
    parser.add_argument('--noise-eval', action='store_true', help='Evaluate policy noise (test if policy is quieter)')
    parser.add_argument('--num-trajectories', type=int, default=10, help='Number of trajectories for noise evaluation (default: 10)')
    parser.add_argument('--x-vel', type=float, default=0.0, help='X velocity command')
    parser.add_argument('--y-vel', type=float, default=0.0, help='Y velocity command')
    parser.add_argument('--yaw-vel', type=float, default=3.14, help='Yaw velocity command')
    parser.add_argument('--logdir', type=str, default='tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--load-model', type=str, help='[Deprecated] Path to .pkl model (use --load-checkpoint-path)')
    parser.add_argument('--load-checkpoint-path', type=str, help='Path to Brax checkpoint directory (logs/.../checkpoints)')
    parser.add_argument('--force-impulse-penalty-scale', type=float, default=None, 
                        help='Override force_impulse_penalty reward scale (default: -0.1, set to 0 to disable)')
    parser.add_argument('--compare-force-csv-a', type=str, default=None,
                        help='Path to first force CSV for policy comparison')
    parser.add_argument('--compare-force-csv-b', type=str, default=None,
                        help='Path to second force CSV for policy comparison')
    parser.add_argument('--compare-output-dir', type=str, default=None,
                        help='Directory to save comparison figures (default: directory of first CSV)')
    parser.add_argument('--compare-label-a', type=str, default='Policy A',
                        help='Label for first policy in comparison figures')
    parser.add_argument('--compare-label-b', type=str, default='Policy B',
                        help='Label for second policy in comparison figures')
    
    args = parser.parse_args()
    
    # If no specific action is requested, run all
    if not any([args.train, args.eval, args.plot, args.interactive, args.noise_eval]):
        args.train = args.eval = args.plot = True
    
    print_section("T1_Force Humanoid Robot with Force Sensors")
    print("The T1_Force is a humanoid robot with force sensors on both feet.")
    
    # Load T1_Force environment
    env_name = 'T1ForceJoystickFlatTerrain'
    env_cfg = registry.get_default_config(env_name)
    
    # Override force_impulse_penalty scale if specified
    if args.force_impulse_penalty_scale is not None:
        env_cfg.reward_config.scales.force_impulse_penalty = args.force_impulse_penalty_scale
        print(f"\nOverriding force_impulse_penalty scale to: {args.force_impulse_penalty_scale}")
    
    env = registry.load(env_name, config=env_cfg)
    
    print(f"\nEnvironment: {env_name}")
    print(f"force_impulse_penalty scale: {env_cfg.reward_config.scales.force_impulse_penalty}")
    print(f"Action size: {env.action_size}")
    print(f"Observation space:")
    obs_shape = env.observation_size
    if isinstance(obs_shape, dict):
        print(f"  State: {obs_shape['state']}")
        print(f"  Privileged state: {obs_shape['privileged_state']}")
    else:
        print(f"  State: {obs_shape}")
    
    # Test force sensors
    test_force_sensors(env)
    
    make_inference_fn = None
    params = None
    metrics = None
    
    # Load existing model if specified
    if args.load_model:
        try:
            make_inference_fn, params, metrics = load_trained_model(args.load_model)
            print("Successfully loaded trained model!")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            return
    elif args.load_checkpoint_path:
        try:
            # Resolve checkpoint path - if it's a checkpoints directory, find latest step
            checkpoint_path = resolve_load_checkpoint_path(args.load_checkpoint_path)
            if checkpoint_path != os.path.normpath(os.path.abspath(os.path.expanduser(args.load_checkpoint_path))):
                print(f"Resolved checkpoint path to: {checkpoint_path}")
            
            # If it's the checkpoints directory, find the latest step checkpoint
            if os.path.isdir(checkpoint_path):
                basename = os.path.basename(checkpoint_path.rstrip('/'))
                
                # If it's a specific step directory (starts with 000), use it directly
                if basename.startswith("000"):
                    resolved_checkpoint_path = checkpoint_path
                    print(f"Using checkpoint at step {basename}")
                # If it's the checkpoints directory, find the latest step
                elif basename == "checkpoints":
                    step_dirs = [
                        d for d in os.listdir(checkpoint_path)
                        if os.path.isdir(os.path.join(checkpoint_path, d)) and d.startswith("000")
                    ]
                    if step_dirs:
                        step_dirs.sort(key=lambda x: int(x))
                        latest_step = step_dirs[-1]
                        resolved_checkpoint_path = os.path.join(checkpoint_path, latest_step)
                        print(f"Found latest checkpoint at step {latest_step}")
                    else:
                        resolved_checkpoint_path = checkpoint_path
                        print(f"Warning: No step directories found in {checkpoint_path}, using directory directly")
                else:
                    # Check if it contains a checkpoints subdirectory
                    checkpoints_subdir = os.path.join(checkpoint_path, "checkpoints")
                    if os.path.exists(checkpoints_subdir):
                        step_dirs = [
                            d for d in os.listdir(checkpoints_subdir)
                            if os.path.isdir(os.path.join(checkpoints_subdir, d)) and d.startswith("000")
                        ]
                        if step_dirs:
                            step_dirs.sort(key=lambda x: int(x))
                            latest_step = step_dirs[-1]
                            resolved_checkpoint_path = os.path.join(checkpoints_subdir, latest_step)
                            print(f"Found latest checkpoint at step {latest_step}")
                        else:
                            resolved_checkpoint_path = checkpoints_subdir
                    else:
                        resolved_checkpoint_path = checkpoint_path
            else:
                resolved_checkpoint_path = checkpoint_path
            
            # Rebuild networks and restore params using ppo.train with restore_checkpoint_path
            ppo_params = locomotion_params.brax_ppo_config(env_name)
            ppo_training_params = dict(ppo_params)
            if 'network_factory' in ppo_training_params:
                del ppo_training_params['network_factory']
            network_factory = ppo_networks.make_ppo_networks
            if hasattr(ppo_params, 'network_factory'):
                network_factory = functools.partial(
                    ppo_networks.make_ppo_networks,
                    **ppo_params.network_factory
                )
            randomizer = registry.get_domain_randomizer(env_name)
            # Force zero training steps by setting num_timesteps=0
            ppo_training_params['num_timesteps'] = 0
            restore_train = functools.partial(
                ppo.train, **dict(ppo_training_params),
                network_factory=network_factory,
                randomization_fn=randomizer,
                restore_checkpoint_path=resolved_checkpoint_path,
                wrap_env_fn=wrapper.wrap_for_brax_training,
            )
            make_inference_fn, params, metrics = restore_train(
                environment=env,
                eval_env=registry.load('T1ForceJoystickFlatTerrain', config=env_cfg),
            )
            print("Successfully restored from checkpoint!")
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Training
    exp_root = None  # Will be set during training or from checkpoint
    if args.train:
        ppo_params, train_fn, writer, ckpt_path, exp_root = setup_training(
            env_name,
            env_cfg,
            logdir=args.logdir,
            enable_tensorboard=not args.no_tensorboard,
        )
        make_inference_fn, params, metrics = train_policy(env, env_cfg, train_fn, writer, ckpt_path)
    
    # Evaluation
    if args.eval and make_inference_fn is not None:
        rollout, eval_data = evaluate_policy(
            env, env_cfg, make_inference_fn, params, 
            args.x_vel, args.y_vel, args.yaw_vel
        )
        
        # Determine unified output dir - prioritize checkpoint's exp_root, then training's exp_root
        output_dir = None
        eval_exp_root = None
        if args.load_checkpoint_path:
            abs_path = resolve_load_checkpoint_path(args.load_checkpoint_path)
            # Case 1: user points directly at an experiment root that contains a 'checkpoints' subdir
            if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "checkpoints")):
                eval_exp_root = abs_path
            # Case 2: path ends with a step dir (e.g., .../checkpoints/000202342400)
            elif os.path.basename(abs_path).startswith("000"):
                checkpoints_dir = os.path.dirname(abs_path)
                eval_exp_root = os.path.dirname(checkpoints_dir)
            # Case 3: path is the 'checkpoints' directory
            elif os.path.basename(abs_path) == "checkpoints":
                eval_exp_root = os.path.dirname(abs_path)
            # Fallback: use the directory itself
            else:
                eval_exp_root = abs_path
            output_dir = eval_exp_root
        elif exp_root:
            # Use training's exp_root if available (when training and evaluating in same run)
            output_dir = exp_root
        else:
            # Fallback to logs directory in current working directory
            output_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output directory for plots/videos/CSV: {output_dir}")

        # Filename suffix from desired velocity command (vx, vy, vyaw)
        eval_suffix = f"vx{args.x_vel:.2f}_vy{args.y_vel:.2f}_vyaw{args.yaw_vel:.2f}"

        # Analyze force_impulse_penalty contribution
        analyze_force_impulse_penalty_contribution(rollout, eval_data, env.dt, force_impulse_scale=-0.1)
        
        # Save force data to CSV
        csv_path = save_force_data_to_csv(rollout, eval_data, env.dt, output_dir, name_suffix=eval_suffix)
        
        if args.plot:
            plot_trajectory_analysis(
                rollout,
                eval_data,
                env.dt,
                output_dir,
                name_suffix=eval_suffix,
                force_impulse_scale=float(
                    env_cfg.reward_config.scales.force_impulse_penalty
                ),
            )
            # Also render the policy
            eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
            render_policy(rollout, eval_data, eval_env, output_dir, name_suffix=eval_suffix)
    
    # Interactive testing
    if args.interactive and make_inference_fn is not None:
        interactive_commands(env, env_cfg, make_inference_fn, params)
    
    # Noise evaluation
    if args.noise_eval and make_inference_fn is not None:
        # Determine output directory for noise evaluation - use same logic as regular eval
        noise_output_dir = None
        if args.load_checkpoint_path:
            abs_path = resolve_load_checkpoint_path(args.load_checkpoint_path)
            # Case 1: user points directly at an experiment root that contains a 'checkpoints' subdir
            if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "checkpoints")):
                exp_root_noise = abs_path
            # Case 2: path ends with a step dir (e.g., .../checkpoints/000202342400)
            elif os.path.basename(abs_path).startswith("000"):
                checkpoints_dir = os.path.dirname(abs_path)
                exp_root_noise = os.path.dirname(checkpoints_dir)
            # Case 3: path is the 'checkpoints' directory
            elif os.path.basename(abs_path) == "checkpoints":
                exp_root_noise = os.path.dirname(abs_path)
            # Fallback: use the directory itself
            else:
                exp_root_noise = abs_path
            noise_output_dir = exp_root_noise
        elif exp_root:
            # Use training's exp_root if available
            noise_output_dir = exp_root
        else:
            # Fallback to logs directory
            noise_output_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(noise_output_dir, exist_ok=True)
        
        print(f"Output directory for noise evaluation: {noise_output_dir}")
        
        average_noise = evaluate_policy_noise(
            env, env_cfg, make_inference_fn, params,
            num_trajectories=args.num_trajectories,
            output_dir=noise_output_dir
        )
        print(f"\nFinal Result: Average episode noise = {average_noise:.4f}")
        print("Use this metric to compare policies - lower values indicate 'quieter' behavior.")

    # Force CSV comparison (does not require env/checkpoints)
    # Policy comparison: pass two force CSVs (same command) to get comparison figures.
    # Outputs: force_Fz_compare.png, velocity_compare.png (if CSV has velocity columns).
    if args.compare_force_csv_a and args.compare_force_csv_b:
        compare_force_csvs(
            args.compare_force_csv_a,
            args.compare_force_csv_b,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
            output_dir=args.compare_output_dir,
        )

if __name__ == "__main__":
    main()
