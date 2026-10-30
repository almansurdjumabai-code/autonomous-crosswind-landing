"""PPO training pipeline for the crosswind landing task (Stable-Baselines3).

Features
--------
* Parallel ``SubprocVecEnv`` rollouts with per-worker ``Monitor`` CSV logs.
* ``VecNormalize`` observation / return normalisation; statistics are saved
  next to every checkpoint and next to the best model.
* Actor-critic MLP with orthogonal initialisation (gain sqrt(2) on hidden
  layers, 0.01 on the policy head, 1.0 on the value head).
* Learning-rate schedule (cosine annealing with linear warm-up, or linear) and
  linearly decaying PPO clip range.
* Wind-severity curriculum: the maximum sampled wind speed ramps from
  ``curriculum_start`` to 1.0 of ``max_wind_speed_mps`` over ``curriculum_steps``.
* TensorBoard logging of PPO diagnostics plus landing metrics (success rate,
  crash rate and reasons, touchdown sink rate / position / crab angle).
* Periodic checkpoints and automatic saving of the best policy by mean
  evaluation return on the full-severity wind distribution.

Usage::

    python -m src.agents.train_ppo --total-timesteps 5000000 --n-envs 8 --run-dir runs/ppo_xwind
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

# Allow running as a plain script (python src/.../file.py) as well as with -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from src.env.aircraft_env import CrosswindLandingEnv, EnvConfig, make_env_config


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class PPOHyperparameters:
    learning_rate: float = 3e-4
    final_learning_rate: float = 1e-5
    lr_schedule: str = "cosine"  # "cosine" | "linear" | "constant"
    lr_warmup_fraction: float = 0.02
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    final_clip_range: float = 0.08
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    log_std_init: float = -0.7
    net_arch_pi: tuple[int, ...] = (256, 256)
    net_arch_vf: tuple[int, ...] = (256, 256)
    activation: str = "tanh"  # "tanh" | "relu" | "elu"


@dataclass
class TrainConfig:
    run_dir: str = "runs/ppo_crosswind"
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    seed: int = 42
    device: str = "auto"
    vec_env: str = "subproc"  # "subproc" | "dummy"
    max_wind_speed_mps: float = 10.0
    curriculum_start: float = 0.2
    curriculum_steps: int = 2_000_000
    eval_freq_steps: int = 50_000
    n_eval_episodes: int = 20
    checkpoint_freq_steps: int = 250_000
    resume_from: Optional[str] = None
    action_mode: str = "residual"  # "residual" (PPO corrects the crosswind autoland) | "absolute"
    residual_scale: float = 0.3
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)


# =============================================================================
# Schedules
# =============================================================================
def linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """SB3 schedule: ``progress_remaining`` goes 1 -> 0 during training."""

    def schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return schedule


def cosine_schedule(initial: float, final: float, warmup_fraction: float = 0.0) -> Callable[[float], float]:
    """Linear warm-up from ``final`` to ``initial`` then cosine annealing to ``final``."""

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if warmup_fraction > 0.0 and progress < warmup_fraction:
            return final + (initial - final) * progress / warmup_fraction
        t = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-12)
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    return schedule


def build_lr_schedule(hp: PPOHyperparameters) -> Callable[[float], float] | float:
    if hp.lr_schedule == "cosine":
        return cosine_schedule(hp.learning_rate, hp.final_learning_rate, hp.lr_warmup_fraction)
    if hp.lr_schedule == "linear":
        return linear_schedule(hp.learning_rate, hp.final_learning_rate)
    if hp.lr_schedule == "constant":
        return hp.learning_rate
    raise ValueError(f"Unknown lr_schedule {hp.lr_schedule!r}")


# =============================================================================
# Environment factories
# =============================================================================
def _make_single_env(rank: int, seed: int, env_config: EnvConfig, monitor_dir: Optional[str]) -> CrosswindLandingEnv:
    env = CrosswindLandingEnv(env_config)
    env.reset(seed=seed + rank)
    env.action_space.seed(seed + rank)
    if monitor_dir is not None:
        env = Monitor(env, filename=os.path.join(monitor_dir, f"worker_{rank:02d}"))
    return env


def make_vec_env(
    n_envs: int, seed: int, env_config: EnvConfig, monitor_dir: Optional[str], kind: str = "subproc"
) -> VecEnv:
    fns = [partial(_make_single_env, i, seed, env_config, monitor_dir) for i in range(n_envs)]
    if kind == "subproc" and n_envs > 1:
        return SubprocVecEnv(fns, start_method="spawn" if platform.system() == "Windows" else "forkserver")
    return DummyVecEnv(fns)


def find_vecnormalize(model_path: str | Path) -> Path:
    """Locate the VecNormalize statistics saved alongside a model by this pipeline.

    Handles ``best/best_model.zip``, ``final_model.zip`` and the checkpoint naming
    ``<prefix>_<steps>_steps.zip`` -> ``<prefix>_vecnormalize_<steps>_steps.pkl``.
    """
    model_path = Path(model_path)
    stem = model_path.stem
    candidates = [model_path.with_name(stem.replace("_model", "_vecnormalize") + ".pkl")]
    if stem.endswith("_steps"):
        prefix, steps = stem[: -len("_steps")].rsplit("_", 1)
        candidates.append(model_path.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"))
    candidates.append(model_path.parent / "vecnormalize.pkl")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No VecNormalize statistics found for {model_path}; tried {[str(c) for c in candidates]}")


def run_action_mode(model_path: str | Path) -> tuple[str, float]:
    """Action mode a checkpoint was trained with, read from the run's ``config.json``.

    Checkpoints from runs that predate the setting were trained with absolute actions.
    """
    for parent in list(Path(model_path).resolve().parents)[:3]:
        cfg = parent / "config.json"
        if cfg.exists():
            train = json.loads(cfg.read_text(encoding="utf-8")).get("train", {})
            return train.get("action_mode", "absolute"), float(train.get("residual_scale", 0.3))
    return "absolute", 0.3


def load_ppo(path: str | Path, env: Optional[VecEnv] = None, device: str = "auto", **kwargs: Any) -> PPO:
    """Load a PPO checkpoint, robust to torch/SB3 version skew.

    ``PPO.load`` hands torch a zip-member stream; some torch releases cannot read
    nested archives from such streams ("PytorchStreamReader failed reading file").
    On that failure this function reproduces SB3's loading procedure with every
    tensor file read into memory first.
    """
    try:
        return PPO.load(str(path), env=env, device=device, **kwargs)
    except RuntimeError as err:
        if "PytorchStreamReader" not in str(err):
            raise
    from stable_baselines3.common.save_util import json_to_data
    from stable_baselines3.common.utils import get_device

    device = get_device(device)
    with zipfile.ZipFile(str(path)) as archive:
        data = json_to_data(archive.read("data").decode(), custom_objects=kwargs.get("custom_objects"))
        params = {
            name[: -len(".pth")]: torch.load(io.BytesIO(archive.read(name)), map_location=device, weights_only=True)
            for name in archive.namelist()
            if name.endswith(".pth") and name != "pytorch_variables.pth"
        }
    data.pop("device", None)
    if env is not None:
        data["n_envs"] = env.num_envs
    model = PPO(policy=data["policy_class"], env=env, device=device, _init_setup_model=False)
    model.__dict__.update(data)
    model.__dict__.update({k: v for k, v in kwargs.items() if k != "custom_objects"})
    model._setup_model()
    model.set_parameters(params, exact_match=True, device=device)
    if env is not None:
        model.env = env
        model._last_obs = None  # force a reset at the start of the next learn()
    return model


# =============================================================================
# Callbacks
# =============================================================================
class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize statistics whenever EvalCallback finds a new best model."""

    def __init__(self, save_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            path = os.path.join(self.save_path, "best_vecnormalize.pkl")
            vec_norm.save(path)
            if self.verbose:
                print(f"[best] saved VecNormalize statistics to {path}")
        return True


class WindCurriculumCallback(BaseCallback):
    """Linearly ramps the wind severity of the *training* environments."""

    def __init__(self, start: float, ramp_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.start = start
        self.ramp_steps = max(int(ramp_steps), 1)
        self._last: float = -1.0

    def severity(self) -> float:
        return min(1.0, self.start + (1.0 - self.start) * self.num_timesteps / self.ramp_steps)

    def _apply(self) -> None:
        s = self.severity()
        if abs(s - self._last) >= 1e-3:
            self.training_env.env_method("set_wind_severity", s)
            self._last = s
        self.logger.record("curriculum/wind_severity", s)

    def _on_training_start(self) -> None:
        self._apply()

    def _on_rollout_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        return True


class LandingMetricsCallback(BaseCallback):
    """Aggregates ``info['episode_metrics']`` over each rollout into TensorBoard scalars."""

    NUMERIC_KEYS = (
        "success",
        "crashed",
        "touched_down",
        "touchdown_sink_rate_mps",
        "touchdown_x_m",
        "touchdown_y_m",
        "touchdown_crab_deg",
        "touchdown_roll_deg",
        "crosswind_20ft_mps",
        "episode_time_s",
    )

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buffer: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if m is not None:
                self._buffer.append(m)
        return True

    def _on_rollout_end(self) -> None:
        if not self._buffer:
            return
        for key in self.NUMERIC_KEYS:
            vals = np.array([float(m[key]) for m in self._buffer], dtype=np.float64)
            if key in ("touchdown_y_m", "touchdown_crab_deg", "touchdown_roll_deg", "crosswind_20ft_mps"):
                vals = np.abs(vals)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                self.logger.record(f"landing/{key}", float(finite.mean()))
        reasons = Counter(m["crash_reason"] for m in self._buffer if m["crash_reason"])
        n = len(self._buffer)
        for reason, count in reasons.items():
            self.logger.record(f"crash_reasons/{reason}", count / n)
        self.logger.record("landing/episodes_in_rollout", n)
        self._buffer.clear()


# =============================================================================
# Training
# =============================================================================
def _activation(name: str) -> type[torch.nn.Module]:
    return {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}[name]


def build_model(cfg: TrainConfig, venv: VecEnv, tb_dir: str) -> PPO:
    hp = cfg.ppo
    policy_kwargs = dict(
        net_arch=dict(pi=list(hp.net_arch_pi), vf=list(hp.net_arch_vf)),
        activation_fn=_activation(hp.activation),
        ortho_init=True,
        log_std_init=hp.log_std_init,
    )
    return PPO(
        "MlpPolicy",
        venv,
        learning_rate=build_lr_schedule(hp),
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        n_epochs=hp.n_epochs,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
        clip_range=linear_schedule(hp.clip_range, hp.final_clip_range),
        ent_coef=hp.ent_coef,
        vf_coef=hp.vf_coef,
        max_grad_norm=hp.max_grad_norm,
        target_kl=hp.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir,
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
    )


def train(cfg: TrainConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    for sub in ("monitor", "checkpoints", "best", "eval", "tb"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    set_random_seed(cfg.seed)

    env_cfg = make_env_config(
        max_wind_speed_mps=cfg.max_wind_speed_mps, action_mode=cfg.action_mode, residual_scale=cfg.residual_scale
    )
    env_cfg.reward = type(env_cfg.reward)(**{**asdict(env_cfg.reward), "gamma": cfg.ppo.gamma})

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "train": asdict(cfg),
                "env": asdict(env_cfg),
                "versions": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                },
            },
            fh,
            indent=2,
            default=str,
        )

    train_env = make_vec_env(cfg.n_envs, cfg.seed, env_cfg, str(run_dir / "monitor"), cfg.vec_env)
    eval_env = make_vec_env(1, cfg.seed + 10_000, env_cfg, None, "dummy")

    if cfg.resume_from:
        resume = Path(cfg.resume_from)
        stats = find_vecnormalize(resume)
        train_env = VecNormalize.load(str(stats), train_env)
        train_env.training = True
        model = load_ppo(resume, env=train_env, device=cfg.device, tensorboard_log=str(run_dir / "tb"))
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.ppo.gamma)
        model = build_model(cfg, train_env, str(run_dir / "tb"))

    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    per_env = lambda steps: max(steps // cfg.n_envs, 1)  # noqa: E731 - callbacks count per-env calls
    callbacks = CallbackList(
        [
            WindCurriculumCallback(cfg.curriculum_start, cfg.curriculum_steps),
            LandingMetricsCallback(),
            CheckpointCallback(
                save_freq=per_env(cfg.checkpoint_freq_steps),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_landing",
                save_vecnormalize=True,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir / "eval"),
                eval_freq=per_env(cfg.eval_freq_steps),
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
                callback_on_new_best=SaveVecNormalizeCallback(str(run_dir / "best"), verbose=1),
                verbose=1,
            ),
        ]
    )

    t0 = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=cfg.resume_from is None,
        progress_bar=False,
    )
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "final_vecnormalize.pkl"))
    print(f"Training finished in {(time.time() - t0) / 60.0:.1f} min. Artifacts in {run_dir.resolve()}")
    train_env.close()
    eval_env.close()
    return run_dir


def parse_args(argv: Optional[list[str]] = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a PPO crosswind-landing policy.")
    d = TrainConfig()
    hp = PPOHyperparameters()
    p.add_argument("--run-dir", default=d.run_dir)
    p.add_argument("--total-timesteps", type=int, default=d.total_timesteps)
    p.add_argument("--n-envs", type=int, default=d.n_envs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--vec-env", choices=("subproc", "dummy"), default=d.vec_env)
    p.add_argument("--max-wind", type=float, default=d.max_wind_speed_mps, help="max wind speed at 20 ft [m/s]")
    p.add_argument("--curriculum-start", type=float, default=d.curriculum_start)
    p.add_argument("--curriculum-steps", type=int, default=d.curriculum_steps)
    p.add_argument("--eval-freq", type=int, default=d.eval_freq_steps)
    p.add_argument("--n-eval-episodes", type=int, default=d.n_eval_episodes)
    p.add_argument("--checkpoint-freq", type=int, default=d.checkpoint_freq_steps)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--action-mode", choices=("residual", "absolute"), default=d.action_mode,
                   help="residual: learn bounded corrections on top of the classical autoland (recommended)")
    p.add_argument("--residual-scale", type=float, default=d.residual_scale)
    p.add_argument("--log-std-init", type=float, default=None,
                   help="initial policy log-std (default -1.6 residual, -0.7 absolute)")
    p.add_argument("--lr", type=float, default=hp.learning_rate)
    p.add_argument("--lr-schedule", choices=("cosine", "linear", "constant"), default=hp.lr_schedule)
    p.add_argument("--n-steps", type=int, default=hp.n_steps)
    p.add_argument("--batch-size", type=int, default=hp.batch_size)
    p.add_argument("--n-epochs", type=int, default=hp.n_epochs)
    p.add_argument("--gamma", type=float, default=hp.gamma)
    p.add_argument("--ent-coef", type=float, default=hp.ent_coef)
    a = p.parse_args(argv)
    ppo = PPOHyperparameters(
        learning_rate=a.lr,
        lr_schedule=a.lr_schedule,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        n_epochs=a.n_epochs,
        gamma=a.gamma,
        ent_coef=a.ent_coef,
        # A residual policy starts from a competent controller: explore gently around it.
        log_std_init=a.log_std_init if a.log_std_init is not None else (-1.6 if a.action_mode == "residual" else -0.7),
    )
    return TrainConfig(
        run_dir=a.run_dir,
        total_timesteps=a.total_timesteps,
        n_envs=a.n_envs,
        seed=a.seed,
        device=a.device,
        vec_env=a.vec_env,
        max_wind_speed_mps=a.max_wind,
        curriculum_start=a.curriculum_start,
        curriculum_steps=a.curriculum_steps,
        eval_freq_steps=a.eval_freq,
        n_eval_episodes=a.n_eval_episodes,
        checkpoint_freq_steps=a.checkpoint_freq,
        resume_from=a.resume_from,
        action_mode=a.action_mode,
        residual_scale=a.residual_scale,
        ppo=ppo,
    )


if __name__ == "__main__":
    train(parse_args())
