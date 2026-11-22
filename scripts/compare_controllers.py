"""Compare the classical, residual, pure PPO and PPO + curriculum controllers on the same seeded episodes.
"""
from argparse import ArgumentParser
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED0 = 10_000  # held out: training evaluation uses 1_000_000+; set i starts at SEED0 * (i + 1)
CHUNK = 20
CONDITIONS = ("nominal", "eval.yaml")


def make_controller(kind, run=None, checkpoint="final", success_dist=0.01):
    from controllers.classical_push import ClassicalPushController

    if kind == "classical":
        return ClassicalPushController(success_dist)

    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from controllers.residual_push import PPOController, ResidualPPOController
    from scripts.eval_ppo import ObservationSpaceEnv, resolve_paths

    _, model_path, stats_path = resolve_paths(run, checkpoint, kind, None, None)
    obs_size = 22 if kind == "residual" else 19
    norm = VecNormalize.load(str(stats_path), DummyVecEnv([lambda: ObservationSpaceEnv(obs_size)]))
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(str(model_path), device="cpu")
    if kind == "residual":
        return ResidualPPOController(model, norm, ClassicalPushController(success_dist))
    return PPOController(model, norm)


def run_chunk(job):
    condition, label, kind, run, checkpoint, episode_set, start, n, e_max = job
    os.chdir(ROOT)
    import gymnasium as gym
    import envs  # noqa: F401  registers PushEnv-v0
    from envs.perturbations import EVAL_CONFIG

    perturbations = EVAL_CONFIG if condition == "eval.yaml" else None
    env = gym.make("PushEnv-v0", perturbations=perturbations, e_max=e_max).unwrapped
    controller = make_controller(kind, run, checkpoint)
    rows = []
    for episode in range(start, start + n):
        controller.reset()
        obs, _ = env.reset(seed=SEED0 * (episode_set + 1) + episode)
        steps = 0
        while True:
            if kind == "classical":
                action = controller.predict(obs)
            else:
                action = controller.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        rows.append((condition, label, episode_set, episode, bool(info["is_success"]), steps))
    env.close()
    return rows


def summarise(rows, groups, n_episodes, n_sets):
    def rate(sel):
        return np.mean([r[4] for r in sel])

    def sd(x):
        return np.nanstd(x, ddof=1) if len(x) > 1 else float("nan")

    def steps(sel):
        ok = [r[5] for r in sel if r[4]]
        return np.mean(ok) if ok else float("nan")

    for condition in CONDITIONS:
        print(f"\n== {condition} ({n_sets} sets x {n_episodes} episodes) ==")
        for label in (lbl for lbls in groups.values() for lbl in lbls):
            per_set = [[r for r in rows if r[0] == condition and r[1] == label and r[2] == i]
                       for i in range(n_sets)]
            print(f"  {label:20} success " + "  ".join(f"{rate(s):6.1%}" for s in per_set)
                  + "   steps " + "  ".join(f"{steps(s):4.0f}" for s in per_set))
        for group, labels in groups.items():
            if not labels:
                continue
            evals = [[r for r in rows if r[0] == condition and r[1] == lbl and r[2] == i]
                     for lbl in labels for i in range(n_sets)]
            s = np.array([rate(e) for e in evals])
            t = np.array([steps(e) for e in evals])
            print(f"  {group:9} over {len(labels)} seed(s) x {n_sets} set(s): "
                  f"success {s.mean():.1%} ± {sd(s):.1%}   steps {np.nanmean(t):.0f} ± {sd(t):.0f}   (mean ± sd)")

        # Steps on the episodes both the classical controller and the learned one solved, so a
        # controller isn't credited for being fast only on the episodes it happens to finish.
        classical = {(r[2], r[3]): r for r in rows if r[0] == condition and r[1] == "classical"}
        for group, labels in groups.items():
            if group == "classical" or not labels:
                continue
            base, ours, ratio, both = [], [], [], []
            for lbl in labels:
                for i in range(n_sets):
                    pairs = [(classical[(i, r[3])][5], r[5]) for r in rows
                             if r[0] == condition and r[1] == lbl and r[2] == i
                             and r[4] and classical[(i, r[3])][4]]
                    both.append(len(pairs))
                    if pairs:
                        c, o = np.array(pairs).T
                        base.append(c.mean())
                        ours.append(o.mean())
                        ratio.extend(1 - o / c)
            if base:
                print(f"  {group:9} vs classical, episodes both solved ({np.mean(both):.0f} of {n_episodes} per eval): "
                      f"steps {np.mean(base):.0f} -> {np.mean(ours):.0f} ± {sd(np.array(ours)):.0f}, "
                      f"median per-episode speedup {np.median(ratio):.0%}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--residual", nargs="*", default=[], help="Residual run directories")
    parser.add_argument("--ppo", nargs="*", default=[], help="Pure PPO run directories")
    parser.add_argument("--ppo-curriculum", nargs="*", default=[], help="Pure PPO + curriculum run directories")
    parser.add_argument(
        "--e-max", type=float,
        help="Clamp the impedance position error to this many metres, for every controller (default: no clamp)",
    )
    parser.add_argument("--checkpoint", choices=("best", "final"), default="final")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--sets", type=int, default=1, help="Disjoint episode sets per controller")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    controllers = [("classical", "classical", None)]
    controllers += [(f"residual {Path(r).name}", "residual", r) for r in args.residual]
    controllers += [(f"ppo {Path(r).name}", "ppo", r) for r in args.ppo]
    controllers += [(f"ppo+curr {Path(r).name}", "ppo", r) for r in args.ppo_curriculum]
    groups = {"classical": ["classical"],
              "residual": [c[0] for c in controllers if c[0].startswith("residual")],
              "ppo": [c[0] for c in controllers if c[0].startswith("ppo ")],
              "ppo+curr": [c[0] for c in controllers if c[0].startswith("ppo+curr")]}

    print(f"impedance error clamp: {args.e_max if args.e_max is not None else 'none'}")
    jobs = [(cond, label, kind, run, args.checkpoint, episode_set, start, min(CHUNK, args.episodes - start), args.e_max)
            for cond in CONDITIONS for label, kind, run in controllers for episode_set in range(args.sets)
            for start in range(0, args.episodes, CHUNK)]
    with Pool(args.workers) as pool:
        rows = [row for chunk in pool.map(run_chunk, jobs) for row in chunk]
    summarise(rows, groups, args.episodes, args.sets)


if __name__ == "__main__":
    main()
