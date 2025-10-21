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
from typing import Dict, List, Tuple

# Add the project root to the Python path to use local mujoco_playground
import os
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
    """Load a trained model from file."""
    import pickle
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded trained model from: {model_path}")
    print(f"Environment: {model_data['env_name']}")
    print(f"Final reward: {model_data['metrics'].get('eval/episode_reward', 'N/A')}")
    
    return model_data['make_inference_fn'], model_data['params'], model_data['metrics']


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


def setup_training(env_name: str, logdir: str = 'tensorboard_logs', enable_tensorboard: bool = True) -> Tuple[Dict, callable, SummaryWriter]:
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
    
    # Setup TensorBoard logging
    writer = None
    if enable_tensorboard:
        log_dir = f"{logdir}/t1_force_joystick"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logging enabled. Logs saved to: {log_dir}")
        print(f"To view training progress, run: tensorboard --logdir {logdir}")
    else:
        print("\nTensorBoard logging disabled.")
    
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
        progress_fn=progress
    )
    
    return ppo_params, train_fn, writer


def train_policy(env, env_cfg, train_fn, writer=None) -> Tuple[callable, Dict, Dict]:
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
    
    # Save trained model
    model_save_dir = "trained_models/t1_force_joystick"
    os.makedirs(model_save_dir, exist_ok=True)
    
    import pickle
    model_path = os.path.join(model_save_dir, "trained_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'make_inference_fn': make_inference_fn,
            'params': params,
            'metrics': metrics,
            'env_name': 'T1ForceJoystickFlatTerrain'
        }, f)
    
    print(f"\nTraining completed!")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Final reward: {metrics.get('eval/episode_reward', 'N/A')}")
    print(f"Trained model saved to: {model_path}")
    
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
    if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
        rng = sample_pert(rng)
    state.info["command"] = command
    
    print(f"Running evaluation with command: Vx={x_vel}, Vy={y_vel}, Vyaw={yaw_vel}")
    
    for i in range(env_cfg.episode_length):
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
        
        if i % 100 == 0:
            print(f"\rStep {i}/{env_cfg.episode_length}", end="")
    
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


def plot_force_analysis(force_data: List, dt: float) -> None:
    """Plot force sensor data analysis."""
    print_section("Force Sensor Analysis")
    
    force_data = jp.array(force_data)
    time_steps = jp.arange(len(force_data)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Left foot forces
    axes[0, 0].plot(time_steps, force_data[:, 0], 'r-', label='Fx')
    axes[0, 0].plot(time_steps, force_data[:, 1], 'g-', label='Fy')
    axes[0, 0].plot(time_steps, force_data[:, 2], 'b-', label='Fz')
    axes[0, 0].set_title('Left Foot Forces')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Force (N)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Right foot forces
    axes[0, 1].plot(time_steps, force_data[:, 3], 'r-', label='Fx')
    axes[0, 1].plot(time_steps, force_data[:, 4], 'g-', label='Fy')
    axes[0, 1].plot(time_steps, force_data[:, 5], 'b-', label='Fz')
    axes[0, 1].set_title('Right Foot Forces')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Vertical forces comparison
    axes[1, 0].plot(time_steps, force_data[:, 2], 'r-', label='Left Fz')
    axes[1, 0].plot(time_steps, force_data[:, 5], 'b-', label='Right Fz')
    axes[1, 0].set_title('Vertical Forces (Ground Reaction)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force (N)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Force magnitude
    left_mag = jp.sqrt(jp.sum(force_data[:, :3]**2, axis=1))
    right_mag = jp.sqrt(jp.sum(force_data[:, 3:]**2, axis=1))
    axes[1, 1].plot(time_steps, left_mag, 'r-', label='Left |F|')
    axes[1, 1].plot(time_steps, right_mag, 'b-', label='Right |F|')
    axes[1, 1].set_title('Force Magnitude')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('|Force| (N)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Print force statistics
    print("Force Sensor Statistics:")
    print("-" * 30)
    print(f"Left foot - Max Fz: {jp.max(force_data[:, 2]):.2f} N")
    print(f"Left foot - Mean Fz: {jp.mean(force_data[:, 2]):.2f} N")
    print(f"Right foot - Max Fz: {jp.max(force_data[:, 5]):.2f} N")
    print(f"Right foot - Mean Fz: {jp.mean(force_data[:, 5]):.2f} N")
    print(f"Total weight support: {jp.mean(force_data[:, 2] + force_data[:, 5]):.2f} N")


def plot_trajectory_analysis(rollout: List, eval_data: Dict, dt: float) -> None:
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
        axes[0].axhline(command[0], color="red", linestyle="--")
        axes[1].axhline(command[1], color="red", linestyle="--")
        axes[2].axhline(command[2], color="red", linestyle="--")

    labels = ["dx", "dy", "dyaw"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])
    
    plt.tight_layout()
    plt.show()

    # Force sensor analysis
    if 'feet_forces' in eval_data and eval_data['feet_forces']:
        plot_force_analysis(eval_data['feet_forces'], dt)

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


def render_policy(rollout: List, eval_data: Dict, eval_env) -> None:
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
    output_path = f"t1_force_policy_render_{int(time.time())}.mp4"
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
    plt.tight_layout()
    plt.show()

    print("Force sensor data shows how the robot adapts its foot forces")
    print("based on different locomotion commands.")


def main():
    """Main function to run T1_Force joystick with force sensors."""
    parser = argparse.ArgumentParser(description='T1_Force Humanoid Locomotion with Force Sensors')
    parser.add_argument('--train', action='store_true', help='Train the policy')
    parser.add_argument('--eval', action='store_true', help='Evaluate the policy')
    parser.add_argument('--plot', action='store_true', help='Plot force sensor analysis')
    parser.add_argument('--interactive', action='store_true', help='Interactive command testing')
    parser.add_argument('--x-vel', type=float, default=0.0, help='X velocity command')
    parser.add_argument('--y-vel', type=float, default=0.0, help='Y velocity command')
    parser.add_argument('--yaw-vel', type=float, default=3.14, help='Yaw velocity command')
    parser.add_argument('--logdir', type=str, default='tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--load-model', type=str, help='Path to trained model to load for evaluation')
    
    args = parser.parse_args()
    
    # If no specific action is requested, run all
    if not any([args.train, args.eval, args.plot, args.interactive]):
        args.train = args.eval = args.plot = True
    
    print_section("T1_Force Humanoid Robot with Force Sensors")
    print("The T1_Force is a humanoid robot with force sensors on both feet.")
    
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
    
    # Training
    if args.train:
        ppo_params, train_fn, writer = setup_training(
            env_name, 
            logdir=args.logdir, 
            enable_tensorboard=not args.no_tensorboard
        )
        make_inference_fn, params, metrics = train_policy(env, env_cfg, train_fn, writer)
    
    # Evaluation
    if args.eval and make_inference_fn is not None:
        rollout, eval_data = evaluate_policy(
            env, env_cfg, make_inference_fn, params, 
            args.x_vel, args.y_vel, args.yaw_vel
        )
        
        if args.plot:
            plot_trajectory_analysis(rollout, eval_data, env.dt)
            # Also render the policy
            eval_env = registry.load('T1ForceJoystickFlatTerrain', config=env_cfg)
            render_policy(rollout, eval_data, eval_env)
    
    # Interactive testing
    if args.interactive and make_inference_fn is not None:
        interactive_commands(env, env_cfg, make_inference_fn, params)

if __name__ == "__main__":
    main()
