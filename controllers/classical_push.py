import numpy as np


class ClassicalPushController:

    STANDOFF = 0.1          # approach point distance behind the block centre
    CONTACT_OFFSET = 0.04   # block centre to pusher tip when touching a face
    LIFT = 0.06             # extra height while approaching from afar
    FREEZE_DIST = 0.02      # keep the push direction fixed this close to the goal
    DT = 0.05               # PushEnv period

    def __init__(self, success_dist=0.01, kp=1.0, ki=0.5):
        self.success_dist = success_dist
        self.kp, self.ki = kp, ki
        self.reset()

    def reset(self):
        self.mode = "approach"
        self.integral = np.zeros(2)
        self.direction = np.array([1.0, 0.0])

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (19,):
            raise ValueError(f"Expected a 19-element observation, got {observation.shape}")

        tip_to_block = observation[2:5]
        block_to_goal = observation[:2]
        goal_distance = np.linalg.norm(block_to_goal)
        goal_direction = block_to_goal / (goal_distance + 1e-8)
        tip_direction = tip_to_block[:2] / (np.linalg.norm(tip_to_block[:2]) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(tip_direction @ goal_direction, -1.0, 1.0)))

        if goal_distance > self.FREEZE_DIST or self.mode == "approach":
            self.direction = goal_direction
        d = np.append(self.direction, 0.0)

        height = np.array([0.0, 0.0, -0.01])
        to_approach = tip_to_block - self.STANDOFF * d + height
        to_contact = tip_to_block - self.CONTACT_OFFSET * d + height

        if self.mode == "approach" and np.linalg.norm(to_approach) < 0.015 and angle < 30:
            self.mode = "push"
        elif angle > 60 and goal_distance > self.success_dist:
            self.mode = "approach"
            self.integral[:] = 0.0
        elif self.mode == "push" and goal_distance < 0.5 * self.success_dist:
            self.mode = "hold"
            self.integral[:] = 0.0
        elif self.mode == "hold" and goal_distance > self.success_dist:
            self.mode = "push"

        if self.mode == "approach":
            action = to_approach.copy()
            if np.linalg.norm(to_approach[:2]) > 0.03:
                action[2] += self.LIFT  # go over the block, not through it
        elif self.mode == "hold":
            action = np.zeros(3)
        else:
            if goal_distance < 0.05:
                self.integral += block_to_goal * self.DT
            pi = self.kp * block_to_goal + self.ki * self.integral
            action = to_contact + np.append(pi, 0.0)

        return action.astype(np.float32)
