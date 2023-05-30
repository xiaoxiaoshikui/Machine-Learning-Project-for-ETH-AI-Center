"""Module for training an RL agent."""
# ruff: noqa: E402
# pylint: disable=wrong-import-position
import os
from os import path
from pathlib import Path

import numpy as np
import torch
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ml_project.reward_model.network import LightningNetwork

ALGORITHM = "ppo"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")
tensorboard_path = path.join(script_path, "..", "..", "rl_logs")
reward_model_path = path.join(
    script_path,
    "..",
    "..",
    "lightning_logs",
    "version_6",
    "checkpoints",
    "epoch=44-step=11610.ckpt",
)


# pylint: disable=too-few-public-methods
class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(self):
        """Initialize custom reward."""
        self.reward_model = LightningNetwork(
            layer_num=3, input_dim=17, hidden_dim=256, output_dim=1
        )
        checkpoint = torch.load(reward_model_path)
        self.reward_model.load_state_dict(checkpoint["state_dict"])
        super().__init__()

    def __call__(
        self,
        state: np.ndarray,
        _action: np.ndarray,
        _next_state: np.ndarray,
        _done: np.ndarray,
    ) -> np.ndarray:
        """Return reward given current state."""
        _batch_idx = 0
        return (
            self.reward_model(torch.Tensor(state), _batch_idx)
            .squeeze()
            .detach()
            .numpy()
        )


def main():
    """Run RL agent training."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    env = make_vec_env(ENVIRONMENT_NAME, n_envs=cpu_count)

    # here we can use the reward model
    if USE_REWARD_MODEL:
        env = RewardVecEnvWrapper(env, reward_fn=CustomReward())

    if ALGORITHM == "sac":
        model = SAC(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path, use_sde=True
        )
    elif ALGORITHM == "ppo":
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_path, use_sde=True
        )
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    iterations = 10

    steps_per_iteration = 5000

    for i in range(iterations):
        model.learn(
            total_timesteps=steps_per_iteration,
            reset_num_timesteps=False,
            log_interval=4,
        )

        model.save(
            path.join(
                models_path,
                f"{ALGORITHM}_{ENVIRONMENT_NAME}_{steps_per_iteration*i}",
            )
        )


if __name__ == "__main__":
    main()
