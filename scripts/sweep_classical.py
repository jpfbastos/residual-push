from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs  # Registers PushEnv-v0
from envs.perturbations import PerturbationConfig

from controllers.classical_push import ClassicalPushController
from scripts.episodes import run_episodes


def main(axis, values, n_episodes=50, seed=0, success_dist=0.01):
    base = PerturbationConfig()  # nominal; only the swept axis changes
    print(f"{axis:>18}  {'success':>8}  {'final_dist':>10}")
    results = []
    for value in values:
        config = base.with_overrides(**{axis: value})
        env = gym.make("PushEnv-v0", success_dist=success_dist, perturbations=config)
        controller = ClassicalPushController(success_dist=success_dist)
        try:
            successes, dists = run_episodes(
                env, controller, controller.predict, n_episodes, seed=seed, verbose=False
            )
        finally:
            env.close()
        results.append((value, float(np.mean(successes)), float(np.nanmean(dists))))
        print(f"{value:>18g}  {results[-1][1]:>8.1%}  {results[-1][2]:>10.4f}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--axis", required=True, help="Perturbation key, e.g. block_mass")
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode k uses seed+k")
    parser.add_argument("--success-dist", type=float, default=0.01)
    args = parser.parse_args()
    main(args.axis, args.values, args.episodes, args.seed, args.success_dist)
