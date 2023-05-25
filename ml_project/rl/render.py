"""Test module for rendering videos of the environment."""

from os import path
from pathlib import Path

import gym
from gym.wrappers import RecordVideo

script_path = Path(__file__).parent.resolve()


def main():
    """Run the rendering test."""
    # Wrap the env by a RecordVideo wrapper
    environment = gym.make("HalfCheetah-v3")
    env = RecordVideo(
        environment,
        video_folder=path.join(script_path, "videos"),
        # episode_trigger=lambda e: True,
        step_trigger=lambda n: n % 500 == 0,
        video_length=100,
    )  # record all episodes

    # Record a video as usual
    _ = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = env.action_space.sample()
        _obs, _reward, done, _info = env.step(action)
        env.render(mode="rgb_array")

    env.close()


if __name__ == "__main__":
    main()
