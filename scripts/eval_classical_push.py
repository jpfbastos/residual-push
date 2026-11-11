from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs

from controllers.classical_push import ClassicalPushController
from scripts.episodes import run_episodes


def main(n_episodes=5, seed=None, render=True, success_dist=0.01):
    env = gym.make(
        "PushEnv-v0",
        render_mode="human" if render else None,
        success_dist=success_dist,
    )
    controller = ClassicalPushController(success_dist=success_dist)

    try:
        run_episodes(env, controller, controller.predict, n_episodes, seed=seed)
    finally:
        env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--success-dist", type=float, default=0.01)
    parser.add_argument(
        "--seed", type=int,
        help="Base seed; episode k uses seed+k. Omit for random episodes",
    )
    parser.add_argument("--no-render", action="store_true", help="Run headless")
    args = parser.parse_args()
    main(args.episodes, args.seed, not args.no_render, args.success_dist)
