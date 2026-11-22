"""Grid search over the classical controller's gains on dedicated tuning episodes.
"""
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED0 = 70_000
CHUNK = 25
# Impedance damping Kd (N s/m, x/y/z) tried at each stiffness K (N/m): light, medium, heavy.
KD_GRID = {
    300.0: ((83.6, 77.7, 104.0), (119.5, 111.0, 148.5), (167.3, 155.4, 207.9)),
    600.0: ((118.3, 109.9, 147.0), (169.0, 157.0, 210.0), (236.6, 219.8, 294.0)),
    900.0: ((144.9, 134.6, 180.0), (207.0, 192.3, 257.2), (289.8, 269.2, 360.1)),
    1200.0: ((167.3, 155.4, 207.9), (239.0, 222.0, 297.0), (334.6, 310.8, 415.8)),
    1800.0: ((204.9, 190.3, 254.6), (292.7, 271.9, 363.7), (409.8, 380.7, 509.2)),
}
STIFFNESSES = tuple(KD_GRID)
KIS = (0.0, 0.5, 1.0, 2.0)
KPS = (1.0,)


def contact_force(env):
    import mujoco
    m, d = env.model, env.data
    pair = {m.geom("block").id, m.geom("pusher").id}
    f6, total = np.zeros(6), 0.0
    for i in range(d.ncon):
        if {d.contact[i].geom1, d.contact[i].geom2} == pair:
            mujoco.mj_contactForce(m, d, i, f6)
            total += float(np.linalg.norm(f6[:3]))
    return total


def run_chunk(job):
    K, kd, kp, ki, start, n, e_max = job
    os.chdir(ROOT)
    import gymnasium as gym
    import envs  # noqa: F401
    from envs.perturbations import EVAL_CONFIG
    from controllers.classical_push import ClassicalPushController

    env = gym.make("PushEnv-v0", perturbations=EVAL_CONFIG, e_max=e_max).unwrapped
    env.ci.Kp = K
    env.ci.Kd = np.array(kd)
    rows = []
    for episode in range(start, start + n):
        controller = ClassicalPushController(kp=kp, ki=ki)
        obs, _ = env.reset(seed=SEED0 + episode)
        steps, peak = 0, 0.0
        while True:
            obs, _, terminated, truncated, info = env.step(controller.predict(obs))
            steps += 1
            peak = max(peak, contact_force(env))
            if terminated or truncated:
                break
        rows.append(dict(K=K, kd=list(kd), kp=kp, ki=ki, episode=episode, success=bool(info["is_success"]),
                         steps=steps, final=float(info["dist"]), peak_contact=peak,
                         off_table=bool(info.get("off_table", False))))
    env.close()
    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--e-max", type=float, help="Impedance error clamp in m (default: none)")
    parser.add_argument("--stiffness", type=float, nargs="+", default=STIFFNESSES)
    parser.add_argument("--kd", nargs="+",
                        help='Impedance damping in N s/m, one number or "x,y,z" per value, '
                             "tried at every stiffness (default: KD_GRID's three settings per stiffness)")
    parser.add_argument("--kp", type=float, nargs="+", default=KPS)
    parser.add_argument("--ki", type=float, nargs="+", default=KIS)
    parser.add_argument("--out", default="logs/tune_classical.json")
    args = parser.parse_args()

    if args.kd:
        kds = [tuple(float(v) for v in k.split(",")) if "," in k else (float(k),) * 3 for k in args.kd]
        pairs = [(K, kd) for K in args.stiffness for kd in kds]
    else:
        missing = [K for K in args.stiffness if K not in KD_GRID]
        if missing:
            parser.error(f"no default Kd for K = {missing}; pass --kd")
        pairs = [(K, kd) for K in args.stiffness for kd in KD_GRID[K]]
    configs = [(K, kd, kp, ki) for (K, kd), kp, ki in product(pairs, args.kp, args.ki)]
    jobs = [(K, kd, kp, ki, s, min(CHUNK, args.episodes - s), args.e_max)
            for K, kd, kp, ki in configs for s in range(0, args.episodes, CHUNK)]
    with Pool(args.workers) as pool:
        rows = [r for chunk in pool.map(run_chunk, jobs) for r in chunk]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))

    summary = []
    for K, kd, kp, ki in configs:
        sel = [r for r in rows if (r["K"], tuple(r["kd"]), r["kp"], r["ki"]) == (K, kd, kp, ki)]
        ok = [r for r in sel if r["success"]]
        summary.append(dict(
            K=K, kd=kd, kp=kp, ki=ki,
            success=np.mean([r["success"] for r in sel]),
            steps=np.mean([r["steps"] for r in ok]) if ok else float("nan"),
            final_cm=np.median([r["final"] for r in sel]) * 100,
            peak_n=np.median([r["peak_contact"] for r in sel]),
            peak90_n=np.percentile([r["peak_contact"] for r in sel], 90),
        ))
    summary.sort(key=lambda s: (-s["success"], s["steps"]))
    print(f"{'K':>6} {'Kd x/y/z (N s/m)':>17} {'kp':>4} {'ki':>4} {'success':>8} {'steps':>6} {'final cm':>9} {'peak N':>7} {'p90 N':>6}")
    for s in summary:
        kd = "/".join(f"{v:.0f}" for v in s["kd"])
        print(f"{s['K']:6.0f} {kd:>17} {s['kp']:4.1f} {s['ki']:4.2f} {s['success']:8.1%} {s['steps']:6.0f} "
              f"{s['final_cm']:9.2f} {s['peak_n']:7.1f} {s['peak90_n']:6.1f}")
    print(f"\n{len(rows)} episodes -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
