#!/usr/bin/env python3
"""Run full force script with force-reward-only environment.

This preserves all features/flags from `t1_joystick_flat_terrain_force.py`
(including `--force-impulse-penalty-scale`, plotting, comparisons, etc.) and
maps the base env name to the force-reward-only env.
"""

import os
import sys

# Add the project root to the Python path to use local mujoco_playground.
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

import t1_joystick_flat_terrain_force as base_force_script


BASE_ENV = "T1ForceJoystickFlatTerrain"
REWARD_ONLY_ENV = "T1ForceJoystickFlatTerrainForceRewardOnly"

_orig_registry_load = registry.load
_orig_registry_default_cfg = registry.get_default_config
_orig_brax_ppo_config = locomotion_params.brax_ppo_config
_orig_setup_training = base_force_script.setup_training


def _map_env(env_name: str) -> str:
  return REWARD_ONLY_ENV if env_name == BASE_ENV else env_name


def _patched_registry_load(env_name, *args, **kwargs):
  return _orig_registry_load(_map_env(env_name), *args, **kwargs)


def _patched_registry_get_default_config(env_name):
  return _orig_registry_default_cfg(_map_env(env_name))


def _patched_brax_ppo_config(env_name, impl=None):
  return _orig_brax_ppo_config(_map_env(env_name), impl=impl)


def _patched_setup_training(env_name, *args, **kwargs):
  return _orig_setup_training(_map_env(env_name), *args, **kwargs)


def main():
  registry.load = _patched_registry_load
  registry.get_default_config = _patched_registry_get_default_config
  locomotion_params.brax_ppo_config = _patched_brax_ppo_config
  base_force_script.setup_training = _patched_setup_training
  base_force_script.main()


if __name__ == "__main__":
  main()
