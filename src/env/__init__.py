"""Crosswind landing environment package; registers ``CrosswindLanding-v0``."""

from gymnasium.envs.registration import register

from src.env.aircraft_env import CrosswindLandingEnv, EnvConfig, make_env_config

ENV_ID = "CrosswindLanding-v0"

register(id=ENV_ID, entry_point="src.env.aircraft_env:CrosswindLandingEnv")

__all__ = ["CrosswindLandingEnv", "EnvConfig", "ENV_ID", "make_env_config"]
