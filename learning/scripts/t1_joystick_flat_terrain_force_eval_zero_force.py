#!/usr/bin/env python3
"""
T1_Force Evaluation Script with Zero Force Input

This script evaluates a trained T1_Force policy while forcing all force sensor inputs to be zero.
This is useful for testing how the policy performs without force sensor feedback.

Usage:
    python t1_joystick_flat_terrain_force_eval_zero_force.py [--plot] [--x-vel X] [--y-vel Y] [--yaw-vel YAW] [--load-checkpoint-path PATH]

Examples:
    # Evaluate with default command (stand still, turn)
    # Can use just the experiment name - script will find it in logs/
    python t1_joystick_flat_terrain_force_eval_zero_force.py --load-checkpoint-path T1ForceJoystickFlatTerrain-20251030_140427
    
    # Or use full path
    python t1_joystick_flat_terrain_force_eval_zero_force.py --load-checkpoint-path logs/T1ForceJoystickFlatTerrain-20251030_140427/checkpoints
    
    # Evaluate with custom command and plot results
    python t1_joystick_flat_terrain_force_eval_zero_force.py --load-checkpoint-path T1ForceJoystickFlatTerrain-20251030_140427 --x-vel 1.0 --y-vel 0.0 --yaw-vel 0.0 --plot
"""

import argparse
import functools
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Add the project root to the Python path to use local mujoco_playground
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
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


def load_trained_model(model_path: str) -> Tuple[callable, Dict, Dict]:
    """Load params and reconstruct make_inference_fn from config/env."""
    # Use cloudpickle to support serializing local/closure functions
    try:
        import cloudpickle as pickle  # type: ignore
    except Exception:
        import pickle  # fallback
    
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


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve checkpoint path by trying multiple locations.
    
    Brax checkpoints can be:
    - A checkpoints directory (e.g., logs/ExperimentName/checkpoints)
    - A specific step directory (e.g., logs/ExperimentName/checkpoints/000106496000)
    - Just the experiment name (will look for logs/ExperimentName/checkpoints)
    
    Tries:
    1. Path as-is (if absolute or exists)
    2. Relative to current working directory
    3. Relative to script directory
    4. In logs/ subdirectory relative to script
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Normalize path
    checkpoint_path = checkpoint_path.rstrip('/')
    
    # Try paths in order
    candidates = [
        checkpoint_path,  # Original path
        os.path.abspath(checkpoint_path),  # Absolute from current dir
        os.path.join(script_dir, checkpoint_path),  # Relative to script
        os.path.join(script_dir, "logs", checkpoint_path),  # In logs/ subdir
    ]
    
    # If path doesn't end with "checkpoints" or a step number, try adding checkpoints
    basename = os.path.basename(checkpoint_path)
    if basename != "checkpoints" and not basename.startswith("000"):
        # Try adding checkpoints subdirectory
        candidates.extend([
            os.path.join(script_dir, "logs", checkpoint_path, "checkpoints"),
            os.path.join(script_dir, checkpoint_path, "checkpoints"),
        ])
    
    # Check each candidate
    for candidate in candidates:
        candidate_abs = os.path.abspath(candidate)
        if os.path.exists(candidate_abs):
            # Verify it's a valid checkpoint directory
            if os.path.isdir(candidate_abs):
                # If it's a specific step directory (starts with 000), return it
                if os.path.basename(candidate_abs).startswith("000"):
                    return candidate_abs
                
                # If it's the checkpoints directory, find the latest step checkpoint
                if os.path.basename(candidate_abs) == "checkpoints":
                    # Find all step directories (those starting with 000)
                    step_dirs = [
                        d for d in os.listdir(candidate_abs)
                        if os.path.isdir(os.path.join(candidate_abs, d)) and d.startswith("000")
                    ]
                    if step_dirs:
                        # Sort by step number (directory name is the step count)
                        step_dirs.sort(key=lambda x: int(x))
                        latest_step = step_dirs[-1]
                        latest_checkpoint = os.path.join(candidate_abs, latest_step)
                        print(f"Found latest checkpoint at step {latest_step}")
                        return latest_checkpoint
                    else:
                        # No step directories found, return the checkpoints dir anyway
                        # (Brax might handle it differently)
                        return candidate_abs
                
                # Check if it contains a checkpoints subdirectory
                checkpoints_subdir = os.path.join(candidate_abs, "checkpoints")
                if os.path.exists(checkpoints_subdir):
                    # Recursively check the checkpoints subdirectory
                    step_dirs = [
                        d for d in os.listdir(checkpoints_subdir)
                        if os.path.isdir(os.path.join(checkpoints_subdir, d)) and d.startswith("000")
                    ]
                    if step_dirs:
                        step_dirs.sort(key=lambda x: int(x))
                        latest_step = step_dirs[-1]
                        latest_checkpoint = os.path.join(checkpoints_subdir, latest_step)
                        print(f"Found latest checkpoint at step {latest_step}")
                        return latest_checkpoint
                    return checkpoints_subdir
    
    # If none found, provide helpful error message
    error_msg = f"Checkpoint path not found: {checkpoint_path}\n"
    error_msg += f"Tried:\n"
    for c in candidates:
        error_msg += f"  - {os.path.abspath(c)}\n"
    error_msg += f"\nAvailable experiments in {os.path.join(script_dir, 'logs')}:\n"
    logs_dir = os.path.join(script_dir, "logs")
    if os.path.exists(logs_dir):
        for item in sorted(os.listdir(logs_dir)):
            item_path = os.path.join(logs_dir, item)
            if os.path.isdir(item_path):
                checkpoints_path = os.path.join(item_path, "checkpoints")
                if os.path.exists(checkpoints_path):
                    error_msg += f"  - {item}/checkpoints\n"
                    # List available step checkpoints
                    step_dirs = [
                        d for d in os.listdir(checkpoints_path)
                        if os.path.isdir(os.path.join(checkpoints_path, d)) and d.startswith("000")
                    ]
                    if step_dirs:
                        step_dirs.sort(key=lambda x: int(x))
                        error_msg += f"    Available steps: {', '.join(step_dirs[-5:])} (showing latest 5)\n"
    raise FileNotFoundError(error_msg)


def zero_force_sensors(obs: Dict) -> Dict:
    """
    Zero out the force sensor inputs in the observation.
    
    The force sensor data is the last 6 elements of the state observation
    (3 per foot: Fx, Fy, Fz for left foot, then right foot).
    """
    if isinstance(obs, dict):
        # Handle dictionary observation format
        modified_obs = {}
        for key, value in obs.items():
            if key == 'state':
                # Force sensors are the last 6 elements of state
                modified_state = jp.array(value)
                modified_state = modified_state.at[-6:].set(0.0)  # Zero last 6 elements
                modified_obs[key] = modified_state
            elif key == 'privileged_state':
                # For privileged state, also zero the force sensors at the end
                modified_priv = jp.array(value)
                # Force sensors are also at the end of privileged state (last 6 elements)
                modified_priv = modified_priv.at[-6:].set(0.0)
                modified_obs[key] = modified_priv
            else:
                modified_obs[key] = value
        return modified_obs
    else:
        # Handle array observation format
        modified_obs = jp.array(obs)
        modified_obs = modified_obs.at[-6:].set(0.0)  # Zero last 6 elements
        return modified_obs


def evaluate_policy_zero_force(env, env_cfg, make_inference_fn, params, 
                                x_vel: float = 0.0, y_vel: float = 0.0, yaw_vel: float = 3.14) -> Tuple[List, List]:
    """Evaluate the trained policy with force sensor inputs forced to zero."""
    print_section("Policy Evaluation (Force Inputs = 0)")
    
    # Setup evaluation environment
    eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
    
    # JIT compile functions for faster execution
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # Create base inference function
    base_inference_fn = make_inference_fn(params, deterministic=True)
    
    # Create wrapped inference function that zeros force sensors
    def zero_force_inference_fn(obs, rng):
        # Zero out force sensor inputs before passing to policy
        modified_obs = zero_force_sensors(obs)
        return base_inference_fn(modified_obs, rng)
    
    jit_inference_fn = jax.jit(zero_force_inference_fn)
    
    print("Evaluation environment setup complete.")
    print("WARNING: Force sensor inputs are forced to zero for this evaluation.")
    
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
    feet_forces = []  # Actual force sensor data (not used by policy)
    obs_forces = []  # Force values that were sent to policy (should be zeros)
    
    command = jp.array([x_vel, y_vel, yaw_vel])
    
    state = jit_reset(rng)
    if has_perturbation_fields(state):
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
    state.info["command"] = command
    
    print(f"Running evaluation with command: Vx={x_vel}, Vy={y_vel}, Vyaw={yaw_vel}")
    print("Force sensor inputs to policy: ZEROED")
    
    for i in range(env_cfg.episode_length):
        if has_perturbation_fields(state):
            if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
                rng = sample_pert(rng)
        act_rng, rng = jax.random.split(rng)
        
        # Inference function already zeros force sensors
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
        
        # Actual force sensor data (for comparison)
        left_force = eval_env.get_left_foot_force(state.data)
        right_force = eval_env.get_right_foot_force(state.data)
        feet_forces.append(jp.hstack([left_force, right_force]))
        
        # Record what was sent to policy (should be zeros)
        modified_obs_for_recording = zero_force_sensors(state.obs)
        if isinstance(modified_obs_for_recording, dict):
            obs_state = modified_obs_for_recording.get('state', jp.zeros(6))
            obs_forces.append(obs_state[-6:])
        else:
            obs_forces.append(modified_obs_for_recording[-6:])
        
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
        
        if i % 100 == 0:
            print(f"\rStep {i}/{env_cfg.episode_length}", end="")
    
    print(f"\nEvaluation completed! Collected {len(rollout)} steps.")
    
    # Verify that force inputs were zeroed
    obs_forces_array = jp.array(obs_forces)
    max_force_input = jp.max(jp.abs(obs_forces_array))
    if max_force_input > 1e-6:
        print(f"WARNING: Force inputs were not properly zeroed! Max value: {max_force_input}")
    else:
        print(f"✓ Verified: Force inputs to policy were zeroed (max: {max_force_input:.2e})")
    
    return rollout, {
        'swing_peak': swing_peak,
        'rewards': rewards,
        'linvel': linvel,
        'angvel': angvel,
        'track': track,
        'foot_vel': foot_vel,
        'rews': rews,
        'contact': contact,
        'feet_forces': feet_forces,  # Actual forces (for comparison)
        'obs_forces': obs_forces,  # Forces sent to policy (should be zeros)
        'modify_scene_fns': modify_scene_fns
    }


def plot_force_analysis(force_data: List, obs_force_data: List, dt: float, output_dir: str = ".") -> None:
    """Plot force sensor data analysis comparing actual vs policy input."""
    print_section("Force Sensor Analysis")
    
    force_data = jp.array(force_data)
    obs_force_data = jp.array(obs_force_data)
    time_steps = jp.arange(len(force_data)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Left foot forces - Actual vs Policy Input
    axes[0, 0].plot(time_steps, force_data[:, 0], 'r-', label='Actual Fx', alpha=0.7)
    axes[0, 0].plot(time_steps, force_data[:, 1], 'g-', label='Actual Fy', alpha=0.7)
    axes[0, 0].plot(time_steps, force_data[:, 2], 'b-', label='Actual Fz', alpha=0.7)
    axes[0, 0].plot(time_steps, obs_force_data[:, 0], 'r--', label='Policy Input Fx (zeroed)', linewidth=2)
    axes[0, 0].plot(time_steps, obs_force_data[:, 1], 'g--', label='Policy Input Fy (zeroed)', linewidth=2)
    axes[0, 0].plot(time_steps, obs_force_data[:, 2], 'b--', label='Policy Input Fz (zeroed)', linewidth=2)
    axes[0, 0].set_title('Left Foot Forces: Actual vs Policy Input')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Force (N)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Right foot forces - Actual vs Policy Input
    axes[0, 1].plot(time_steps, force_data[:, 3], 'r-', label='Actual Fx', alpha=0.7)
    axes[0, 1].plot(time_steps, force_data[:, 4], 'g-', label='Actual Fy', alpha=0.7)
    axes[0, 1].plot(time_steps, force_data[:, 5], 'b-', label='Actual Fz', alpha=0.7)
    axes[0, 1].plot(time_steps, obs_force_data[:, 3], 'r--', label='Policy Input Fx (zeroed)', linewidth=2)
    axes[0, 1].plot(time_steps, obs_force_data[:, 4], 'g--', label='Policy Input Fy (zeroed)', linewidth=2)
    axes[0, 1].plot(time_steps, obs_force_data[:, 5], 'b--', label='Policy Input Fz (zeroed)', linewidth=2)
    axes[0, 1].set_title('Right Foot Forces: Actual vs Policy Input')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Vertical forces comparison
    axes[1, 0].plot(time_steps, force_data[:, 2], 'r-', label='Actual Left Fz', alpha=0.7)
    axes[1, 0].plot(time_steps, force_data[:, 5], 'b-', label='Actual Right Fz', alpha=0.7)
    axes[1, 0].plot(time_steps, obs_force_data[:, 2], 'r--', label='Policy Input Left Fz (zeroed)', linewidth=2)
    axes[1, 0].plot(time_steps, obs_force_data[:, 5], 'b--', label='Policy Input Right Fz (zeroed)', linewidth=2)
    axes[1, 0].set_title('Vertical Forces: Actual vs Policy Input')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force (N)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Force magnitude comparison
    left_mag_actual = jp.sqrt(jp.sum(force_data[:, :3]**2, axis=1))
    right_mag_actual = jp.sqrt(jp.sum(force_data[:, 3:]**2, axis=1))
    left_mag_policy = jp.sqrt(jp.sum(obs_force_data[:, :3]**2, axis=1))
    right_mag_policy = jp.sqrt(jp.sum(obs_force_data[:, 3:]**2, axis=1))
    axes[1, 1].plot(time_steps, left_mag_actual, 'r-', label='Actual Left |F|', alpha=0.7)
    axes[1, 1].plot(time_steps, right_mag_actual, 'b-', label='Actual Right |F|', alpha=0.7)
    axes[1, 1].plot(time_steps, left_mag_policy, 'r--', label='Policy Input Left |F| (zeroed)', linewidth=2)
    axes[1, 1].plot(time_steps, right_mag_policy, 'b--', label='Policy Input Right |F| (zeroed)', linewidth=2)
    axes[1, 1].set_title('Force Magnitude: Actual vs Policy Input')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('|Force| (N)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"force_analysis_zeroed_{int(time.time())}.png")
    fig.savefig(fig_path, dpi=150)
    plt.show()

    # Print force statistics
    print("Force Sensor Statistics:")
    print("-" * 30)
    print(f"Actual Left foot - Max Fz: {jp.max(force_data[:, 2]):.2f} N")
    print(f"Actual Left foot - Mean Fz: {jp.mean(force_data[:, 2]):.2f} N")
    print(f"Actual Right foot - Max Fz: {jp.max(force_data[:, 5]):.2f} N")
    print(f"Actual Right foot - Mean Fz: {jp.mean(force_data[:, 5]):.2f} N")
    print(f"Policy Input - Max absolute value: {jp.max(jp.abs(obs_force_data)):.2e} N (should be ~0)")


def plot_trajectory_analysis(rollout: List, eval_data: Dict, dt: float, output_dir: str = ".") -> None:
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
        fig_path = os.path.join(output_dir, f"swing_peaks_zeroed_{int(time.time())}.png")
        fig.savefig(fig_path, dpi=150)
        plt.show()

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
        axes[0].axhline(command[0], color="red", linestyle="--", label="Command")
        axes[1].axhline(command[1], color="red", linestyle="--", label="Command")
        axes[2].axhline(command[2], color="red", linestyle="--", label="Command")

    labels = ["dx", "dy", "dyaw"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])
        ax.legend()
    
    plt.tight_layout()
    # Save velocity tracking figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path2 = os.path.join(output_dir, f"velocity_tracking_zeroed_{int(time.time())}.png")
    fig.savefig(fig_path2, dpi=150)
    plt.show()

    # Force sensor analysis
    if 'feet_forces' in eval_data and 'obs_forces' in eval_data and eval_data['feet_forces']:
        plot_force_analysis(eval_data['feet_forces'], eval_data['obs_forces'], dt, output_dir)

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
        print("\nNote: This evaluation was run with force sensor inputs forced to zero.")


def render_policy(rollout: List, eval_data: Dict, eval_env, output_dir: str = ".") -> None:
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
    output_path = os.path.join(output_dir, f"t1_force_policy_render_zeroed_{int(time.time())}.mp4")
    media.write_video(output_path, frames, fps=fps)
    print(f"Video saved to: {output_path}")


def main():
    """Main function to run T1_Force evaluation with zero force inputs."""
    parser = argparse.ArgumentParser(description='T1_Force Evaluation with Zero Force Inputs')
    parser.add_argument('--eval', action='store_true', default=True, help='Evaluate the policy (default)')
    parser.add_argument('--plot', action='store_true', help='Plot force sensor analysis')
    parser.add_argument('--x-vel', type=float, default=0.0, help='X velocity command')
    parser.add_argument('--y-vel', type=float, default=0.0, help='Y velocity command')
    parser.add_argument('--yaw-vel', type=float, default=3.14, help='Yaw velocity command')
    parser.add_argument('--load-model', type=str, help='[Deprecated] Path to .pkl model (use --load-checkpoint-path)')
    parser.add_argument('--load-checkpoint-path', type=str, required=True, 
                       help='Path to Brax checkpoint directory. Can be: '
                            'experiment name (e.g., T1ForceJoystickFlatTerrain-20251030_140427), '
                            'relative path (e.g., logs/T1ForceJoystickFlatTerrain-20251030_140427/checkpoints), '
                            'or absolute path. The script will automatically search in the logs/ directory.')
    
    args = parser.parse_args()
    
    print_section("T1_Force Evaluation with Zero Force Inputs")
    print("This script evaluates a trained T1_Force policy while forcing")
    print("all force sensor inputs to zero to test policy performance without force feedback.")
    
    # Load T1_Force environment
    env_name = 'T1ForceJoystickFlatTerrain'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    
    print(f"\nEnvironment: {env_name}")
    print(f"Action size: {env.action_size}")
    print(f"Observation space:")
    obs_shape = env.observation_size
    if isinstance(obs_shape, dict):
        print(f"  State: {obs_shape['state']}")
        print(f"  Privileged state: {obs_shape['privileged_state']}")
    else:
        print(f"  State: {obs_shape}")
    
    make_inference_fn = None
    params = None
    metrics = None
    resolved_checkpoint_path = None  # Store resolved path for output directory
    
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
            # Resolve checkpoint path (try multiple locations)
            resolved_checkpoint_path = resolve_checkpoint_path(args.load_checkpoint_path)
            print(f"Resolved checkpoint path: {resolved_checkpoint_path}")
            
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
    
    # Evaluation
    if args.eval and make_inference_fn is not None:
        rollout, eval_data = evaluate_policy_zero_force(
            env, env_cfg, make_inference_fn, params, 
            args.x_vel, args.y_vel, args.yaw_vel
        )
        
        # Determine unified output dir
        output_dir = None
        if resolved_checkpoint_path:
            abs_path = os.path.abspath(resolved_checkpoint_path)
            # If path is a step directory (starts with 000), go up one level to checkpoints, then up to exp_root
            if os.path.basename(abs_path).startswith("000"):
                checkpoints_dir = os.path.dirname(abs_path)
                exp_root = os.path.dirname(checkpoints_dir)
            else:
                # Path is the checkpoints directory itself
                exp_root = os.path.dirname(abs_path)
            output_dir = exp_root
        else:
            # Fallback to current directory
            output_dir = os.getcwd()

        if args.plot:
            plot_trajectory_analysis(rollout, eval_data, env.dt, output_dir)
            # Also render the policy
            eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
            render_policy(rollout, eval_data, eval_env, output_dir)

if __name__ == "__main__":
    main()

