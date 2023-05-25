"""Module for training a reward model from trajectory data."""
import pickle
import random
from os import path
from pathlib import Path

import numpy as np

from ..types import Batch, Trajectories
from .network import Network

current_path = Path(__file__).parent.resolve()
folder_path = path.join(current_path, "../rl/reward_data")
file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")


def sample_preference_batch(trajectories: Trajectories, batch_size=32):
    """
    Create a random batch consisting of tuples of observations from two trajectories.

    The tuples are sorted based on the (undiscounted) cumulative reward of the
    trajectories: (obs0, obs1) iff reward1 > reward0.
    """
    batch: Batch = []
    for _ in range(batch_size):
        indices = random.sample(list(trajectories.keys()), 2)

        reward0 = np.sum([entry["reward"] for entry in trajectories[indices[0]]])
        obs0 = [entry["obs"] for entry in trajectories[indices[0]]]

        reward1 = np.sum([entry["reward"] for entry in trajectories[indices[1]]])
        obs1 = [entry["obs"] for entry in trajectories[indices[1]]]

        if reward1 > reward0:
            batch.append((obs0, obs1))
        else:
            batch.append((obs1, obs0))

    return batch


def main():
    with open(file_name, "rb") as handle:
        trajectories = pickle.load(handle)

    batch = sample_preference_batch(trajectories, batch_size=1)
    input_dim = np.array(batch[0][0][0]).shape

    Network(layer_num=3, input_dim=input_dim, hidden_dim=256, output_dim=1)

    # train model using loss in https://arxiv.org/pdf/1904.06387.pdf


if __name__ == "__main__":
    main()
