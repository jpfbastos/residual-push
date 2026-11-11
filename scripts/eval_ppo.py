from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
from gymnasium import spaces
import envs  # Registers PushEnv-v0
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from controllers.classical_push import ClassicalPushController
from controllers.residual_push import PPOController, ResidualPPOController
from scripts.episodes import run_episodes


class ObservationSpaceEnv(gym.Env):
    metadata = {"render_modes": []}
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def __init__(self, observation_size):
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("ObservationSpaceEnv is only used to load statistics")


def resolve_paths(run, checkpoint, mode, model_path, stats_path):
    """Pick model/stats from a run directory, unless given explicitly."""
    if run is not None:
        run = Path(run)
        args_file = run / "args.json"
        if mode is None and args_file.exists():
            mode = json.loads(args_file.read_text())["mode"]
        if checkpoint == "best":
            model_path = model_path or run / "best" / "best_model.zip"
            stats_path = stats_path or run / "best" / "vecnormalize.pkl"
        else:
            model_path = model_path or run / "final_model.zip"
            stats_path = stats_path or run / "final_vecnormalize.pkl"
    if model_path is None or stats_path is None:
        raise SystemExit("Pass --run, or both --model and --stats")
    return mode or "ppo", Path(model_path), Path(stats_path)


def main(
    mode=None,
    run=None,
    checkpoint="best",
    stats_path=None,
    model_path=None,
    n_episodes=5,
    success_dist=0.01,
    seed=None,
    render=True,
):
    mode, model_path, stats_path = resolve_paths(run, checkpoint, mode, model_path, stats_path)
    if mode not in {"ppo", "residual"}:
        raise ValueError(f"Unknown PPO mode: {mode}")
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find environment statistics: {stats_path}")
    print(f"mode={mode}  model={model_path}")

    observation_size = 19 if mode == "ppo" else 22
    stats_env = VecNormalize.load(
        str(stats_path),
        DummyVecEnv([lambda: ObservationSpaceEnv(observation_size)]),
    )
    stats_env.training = False
    stats_env.norm_reward = False

    model = PPO.load(str(model_path))
    if mode == "ppo":
        controller = PPOController(model, normalizer=stats_env)
    else:
        classical = ClassicalPushController(success_dist=success_dist)
        controller = ResidualPPOController(
            model,
            normalizer=stats_env,
            classical_controller=classical,
        )

    env = gym.make(
        "PushEnv-v0",
        render_mode="human" if render else None,
        success_dist=success_dist,
    )
    try:
        run_episodes(
            env,
            controller,
            lambda obs: controller.predict(obs, deterministic=True),
            n_episodes,
            seed=seed,
        )
    finally:
        env.close()
        stats_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", help="Run directory, e.g. runs/residual/20261111-120000_seed42")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument(
        "--mode", choices=("ppo", "residual"),
        help="Defaults to the mode in the run's args.json, else ppo",
    )
    parser.add_argument("--stats", help="VecNormalize statistics file (overrides --run)")
    parser.add_argument("--model", help="PPO checkpoint (overrides --run)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--success-dist", type=float, default=0.01)
    parser.add_argument(
        "--seed", type=int,
        help="Base seed; episode k uses seed+k. Omit for random episodes",
    )
    parser.add_argument("--no-render", action="store_true", help="Run headless")
    args = parser.parse_args()
    main(
        mode=args.mode,
        run=args.run,
        checkpoint=args.checkpoint,
        stats_path=args.stats,
        model_path=args.model,
        n_episodes=args.episodes,
        success_dist=args.success_dist,
        seed=args.seed,
        render=not args.no_render,
    )
