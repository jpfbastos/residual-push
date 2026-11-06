from argparse import ArgumentParser
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


def main(mode, stats_path, model_path=None, n_episodes=5, success_dist=0.01):
    if mode not in {"ppo", "residual"}:
        raise ValueError(f"Unknown PPO mode: {mode}")

    default_model = (
        ROOT / "models" / "best" / "best_model.zip"
        if mode == "ppo"
        else ROOT / "models" / "residual" / "best" / "best_model.zip"
    )
    model_path = Path(model_path) if model_path else default_model
    stats_path = Path(stats_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find environment statistics: {stats_path}")

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
        "PushEnv-v0", render_mode="human", success_dist=success_dist
    )
    try:
        for episode in range(n_episodes):
            controller.reset()
            observation, _ = env.reset()
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = controller.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            print(
                f"Episode {episode}: reward={total_reward:.3f}  "
                f"success={info.get('is_success')}  "
                f"off_table={info.get('off_table')}  "
                f"truncated={truncated}  "
                f"final_dist={info.get('dist'):.4f}"
            )
    finally:
        env.close()
        stats_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=("ppo", "residual"), default="ppo")
    parser.add_argument("--stats", required=True, help="VecNormalize statistics file")
    parser.add_argument("--model", help="PPO checkpoint (defaults by mode)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--success-dist", type=float, default=0.01)
    args = parser.parse_args()
    main(args.mode, args.stats, args.model, args.episodes, args.success_dist)
