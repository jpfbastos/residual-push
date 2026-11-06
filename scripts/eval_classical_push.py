from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym, numpy as np
import envs

env = gym.make("PushEnv-v0", render_mode="human", success_dist=0.01)

observation, info = env.reset()

terminated = False
truncated = False

action = env.action_space.sample()

pushing = False

while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(action)

    # vector between block -> goal (xy):       0
    # vector between pusher -> block (xyz):    2
    # block pose yaw as sin/cos pair:          5
    # block linear velocity (xy):              7
    # block yaw rate:                          9
    # pusher state absolute position (xyz):    10
    # pusher state absolute velocity (xyz):    13
    # previous action (xyz):                   16

    tip_to_block = obs[2:5]
    tip_to_block_dist = np.linalg.norm(tip_to_block)
    tip_to_block_normalised = tip_to_block / tip_to_block_dist
    block_to_goal = np.append(obs[0:2], 0)
    block_to_goal_dist = np.linalg.norm(block_to_goal)
    block_to_goal_normalised = block_to_goal / block_to_goal_dist

    angle = np.arccos(np.clip(np.dot(tip_to_block_normalised, block_to_goal_normalised), -1.0, 1.0))

    if not pushing:
        action = tip_to_block - block_to_goal_normalised * 0.1
        pushing = tip_to_block[2] > -0.01 and np.linalg.norm(action) < 0.1 and angle < np.pi / 5
    else:
        if block_to_goal_dist > env.env.env.success_dist:
            action = tip_to_block + block_to_goal
            pushing = angle < np.pi / 4
        else:
            action = np.zeros(3)

env.close()