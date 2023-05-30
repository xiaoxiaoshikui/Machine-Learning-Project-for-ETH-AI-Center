"""Module for saving videos and data of an RL agent's trajectories."""
import os
import pickle
from os import path
from pathlib import Path
from typing import Type, Union

import gym
from gym.wrappers import RecordVideo
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ..types import Step, Trajectories

ALGORITHM = "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"

RECORD_INTERVAL = 500
RECORD_LENGTH = 100
VIDEOS_PER_CHECKPOINT = 5

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models")


def record_videos(
    algorithm: Union[Type[PPO], Type[SAC]], environment: gym.wrappers.RecordVideo
):
    """Record videos of the training environment."""
    obs_dataset: Trajectories = {}
    observations: list[Step] = []

    n_step = 0
    # max_videos = 20  # Define your maximum videos here
    video_count = 0

    for n_checkpoint, file in enumerate(os.listdir(models_path)):
        if file.startswith(ALGORITHM + "_" + ENVIRONMENT_NAME):
            model = algorithm.load(path.join(models_path, file[:-4]))

            obs = environment.reset()
            while n_step < (n_checkpoint + 1) * VIDEOS_PER_CHECKPOINT * RECORD_INTERVAL:
                action, _states = model.predict(obs, deterministic=True)
                obs, _reward, terminated, _info = environment.step(action)

                environment.render(mode="rgb_array")

                i = n_step % RECORD_INTERVAL
                if i < RECORD_LENGTH:
                    if i == 0:
                        observations = []

                    observations.append(obs)

                    if i == RECORD_LENGTH - 1:
                        obs_dataset[n_step - RECORD_LENGTH + 1] = observations
                        video_count += 1  # Increment the video count

                # Break the loop if maximum videos have been recorded
                # if video_count >= max_videos:
                #    print(f"Stopped recording after {video_count} videos.")
                #    return obs_dataset

                if terminated:
                    obs = environment.reset()

                n_step += 1

    return obs_dataset


def main():
    """Run video generation."""
    env = gym.make(ENVIRONMENT_NAME)
    env = RecordVideo(
        env,
        video_folder=path.join(script_path, "..", "static", "sac_videos"),
        step_trigger=lambda n: n % RECORD_INTERVAL == 0,
        video_length=RECORD_LENGTH,
        name_prefix=f"{ALGORITHM}-{ENVIRONMENT_NAME}",
    )

    if ALGORITHM == "sac":
        ALGO = SAC
    elif ALGORITHM == "ppo":
        ALGO = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    obs_dataset = record_videos(ALGO, env)

    with open(
        path.join(
            script_path,
            "observation_data_corresponding_to_videos",
            f"{ALGORITHM}_{ENVIRONMENT_NAME}_obs_dataset.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(obs_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
