"""Shared episode loop for the evaluation scripts.

With ``seed`` given, episode k is reset with seed ``seed + k`` (reproducible).
With ``seed=None``, episodes are plain random.
Since starting conditions require sampling from seed in a while loop, subsequent samples
depend on previous samples, so using a different seed per episode means environments will stay synced
"""
import numpy as np


def run_episodes(env, controller, predict, n_episodes, seed=None):
    successes, final_dists = [], []
    for episode in range(n_episodes):
        episode_seed = None if seed is None else seed + episode
        controller.reset()
        observation, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        total_reward = 0.0
        info = {}

        while not (terminated or truncated):
            action = predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        successes.append(bool(info.get("is_success")))
        final_dists.append(float(info.get("dist", np.nan)))
        print(
            f"Episode {episode}: reward={total_reward:.3f}  "
            f"success={info.get('is_success')}  "
            f"off_table={info.get('off_table')}  "
            f"truncated={truncated}  "
            f"final_dist={final_dists[-1]:.4f}"
        )

    print(
        f"\n{n_episodes} episodes: "
        f"success={np.mean(successes):.2%}  "
        f"mean final_dist={np.nanmean(final_dists):.4f}"
    )
    return successes, final_dists
