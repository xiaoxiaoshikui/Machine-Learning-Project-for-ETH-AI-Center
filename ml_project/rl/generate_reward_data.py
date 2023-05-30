"""Module for saving data of an RL agent's trajectories after training."""
import os
import pickle
from os import path
from pathlib import Path
from typing import Type, Union

import gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

ALGORITHM = "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"

STEPS_PER_CHECKPOINT = 10000

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")


def generate_data(algorithm: Union[Type[PPO], Type[SAC]], environment: gym.Env):
    """Generate agent's observations and rewards in the training environment."""
    data = []

    for file in os.listdir(models_path):
        if file.startswith(ALGORITHM + "_" + ENVIRONMENT_NAME):
            model = algorithm.load(path.join(models_path, file[:-4]))

            n_step = 0
            obs = environment.reset()
            while n_step < STEPS_PER_CHECKPOINT:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, _info = environment.step(action)

                data.append({"obs": obs, "reward": reward})

                if terminated:
                    obs = environment.reset()

                n_step += 1

    return data


def main():
    """Run data generation."""
    env = gym.make(ENVIRONMENT_NAME)

    if ALGORITHM == "sac":
        ALGO = SAC
    elif ALGORITHM == "ppo":
        ALGO = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    obs_reward_data = generate_data(ALGO, env)

    with open(
        path.join(
            script_path,
            "reward_data",
            f"{ALGORITHM}_{ENVIRONMENT_NAME}_obs_reward_dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(obs_reward_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
