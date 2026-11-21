import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


ACTION_LIMIT = 1.0
RESIDUAL_SCALE = 0.1  # with the initial policy std of ~0.74, ~7.4 cm of exploration


class PPOController:
    def __init__(
        self,
        model: PPO,
        normalizer: VecNormalize | None = None,
    ):
        self.model = model
        self.normalizer = normalizer

    def reset(self):
        pass

    def _predict_policy(self, policy_observation, deterministic):
        if self.normalizer is not None:
            policy_observation = self.normalizer.normalize_obs(
                policy_observation[np.newaxis, :]
            )[0]
        action, _ = self.model.predict(
            policy_observation, deterministic=deterministic
        )
        return np.asarray(action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (19,):
            raise ValueError(f"Expected a 19-element observation, got {observation.shape}")
        action = self._predict_policy(observation, deterministic)
        if action.shape != (3,):
            raise ValueError(f"Expected a 3-element PPO action, got {action.shape}")
        return np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)


class ResidualPPOController(PPOController):
    def __init__(
        self,
        model: PPO,
        normalizer: VecNormalize | None = None,
        classical_controller=None,
    ):
        super().__init__(model, normalizer)
        if classical_controller is None:
            from controllers.classical_push import ClassicalPushController

            classical_controller = ClassicalPushController()
        self.classical_controller = classical_controller

    def reset(self):
        self.classical_controller.reset()

    def predict(self, observation, deterministic=True):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (19,):
            raise ValueError(f"Expected a 19-element observation, got {observation.shape}")

        classical_action = self.classical_controller.predict(observation)
        policy_observation = np.concatenate((observation, classical_action))
        delta = self._predict_policy(policy_observation, deterministic)
        if delta.shape != (3,):
            raise ValueError(f"Expected a 3-element PPO action, got {delta.shape}")
        residual = RESIDUAL_SCALE * np.clip(delta, -1.0, 1.0)
        return np.clip(classical_action + residual, -ACTION_LIMIT, ACTION_LIMIT)