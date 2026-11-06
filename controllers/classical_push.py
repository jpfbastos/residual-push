import numpy as np


class ClassicalPushController:
    def __init__(self, success_dist=0.1):
        self.success_dist = success_dist
        self.reset()

    def reset(self):
        self.pushing = False

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (19,):
            raise ValueError(f"Expected a 19-element observation, got {observation.shape}")

        tip_to_block = observation[2:5]
        block_to_goal = np.append(observation[:2], 0.0)
        goal_distance = np.linalg.norm(block_to_goal)
        goal_direction = block_to_goal / (goal_distance + 1e-8)
        tip_direction = tip_to_block / (np.linalg.norm(tip_to_block) + 1e-8)
        angle = np.arccos(np.clip(np.dot(tip_direction, goal_direction), -1.0, 1.0))

        if not self.pushing:
            action = tip_to_block - goal_direction * 0.1
            self.pushing = (
                tip_to_block[2] > -0.01
                and np.linalg.norm(action) < 0.1
                and angle < np.pi / 5
            )
        elif goal_distance > self.success_dist:
            action = tip_to_block + block_to_goal
            self.pushing = angle < np.pi / 4
        else:
            action = np.zeros(3)

        return action.astype(np.float32)


