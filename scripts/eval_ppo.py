from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs  # Registers PushEnv-v0
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


def main(n_episodes=5):
    model_path = ROOT / "models" / "best" / "best_model.zip"
    stats_path = ROOT / "models" / "best" / "vecnormalize.pkl"

    for p in (model_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"Could not find: {p}")

    venv = make_vec_env(
        "PushEnv-v0",
        n_envs=1,
        env_kwargs=dict(render_mode="human", success_dist=0.01),
    )
    venv = VecNormalize.load(str(stats_path), venv)
    venv.training = False        # freeze running stats
    venv.norm_reward = False     # report real rewards

    model = PPO.load(str(model_path), env=venv)

    try:
        for ep in range(n_episodes):
            obs = venv.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = venv.step(action)   # batched; auto-resets on done
                total_reward += float(reward[0])
                done = bool(dones[0])

            info = infos[0]    # info from the final step of the episode
            print(f"Episode {ep}: reward={total_reward:.3f}  "
                  f"success={info.get('is_success')}  off_table={info.get('off_table')}  "
                  f"truncated={info.get('TimeLimit.truncated', False)}  "
                  f"final_dist={info.get('dist'):.4f}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()