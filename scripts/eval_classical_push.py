from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs

from controllers.classical_push import ClassicalPushController


def main(n_episodes=5):
    env = gym.make(
        "PushEnv-v0", render_mode="human", success_dist=0.01
    )
    controller = ClassicalPushController(success_dist=0.01)

    try:
        for episode in range(n_episodes):
            controller.reset()
            observation, _ = env.reset()
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = controller.predict(observation)
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


if __name__ == "__main__":
    main()
