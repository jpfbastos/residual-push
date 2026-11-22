from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs
from envs.perturbations import EVAL_CONFIG
import numpy as np

from controllers.classical_push import ClassicalPushController
from scripts.episodes import run_episodes


def main(n_episodes=5, seed=None, render=True, success_dist=0.01, kp=1.0, ki=0.5, stiffness=None, kd=None, e_max=None):
    print(f"perturbations={EVAL_CONFIG.relative_to(ROOT)}  kp={kp}  ki={ki}  stiffness={stiffness or 'default'}  kd={kd or 'default'}  e_max={e_max}")
    env = gym.make(
        "PushEnv-v0",
        render_mode="human" if render else None,
        success_dist=success_dist,
        perturbations=EVAL_CONFIG,
        e_max=e_max,
    )
    if stiffness is not None:
        env.unwrapped.ci.Kp = stiffness
    if kd is not None:
        env.unwrapped.ci.Kd = np.array(kd if len(kd) == 3 else kd * 3)
    controller = ClassicalPushController(success_dist=success_dist, kp=kp, ki=ki)

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
    parser.add_argument("--kp", type=float, default=1.0, help="Outer-loop proportional gain")
    parser.add_argument("--ki", type=float, default=0.5, help="Outer-loop integral gain (0 for P only)")
    parser.add_argument(
        "--stiffness", type=float,
        help="Impedance stiffness K in N/m (default 900)",
    )
    parser.add_argument(
        "--kd", type=float, nargs="+",
        help="Impedance damping in N s/m: one value for all axes or three for x y z (default 289.8 269.2 360.1)",
    )
    parser.add_argument(
        "--e-max", type=float,
        help="Clamp the impedance position error to this many metres (default: no clamp)",
    )
    args = parser.parse_args()
    main(args.episodes, args.seed, not args.no_render, args.success_dist,
         kp=args.kp, ki=args.ki, stiffness=args.stiffness, kd=args.kd, e_max=args.e_max)
