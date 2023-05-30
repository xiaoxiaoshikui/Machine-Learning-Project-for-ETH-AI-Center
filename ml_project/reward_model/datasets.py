"""Module containing the Datasets required for neural network training."""
import pickle
from itertools import chain

import numpy
from torch.utils.data import Dataset

from ..types import FloatNDArray, RewardlessTrajectory


class PreferenceDataset(Dataset):
    """PyTorch Dataset for loading preference data."""

    def __init__(self, dataset_path: str):
        """Initialize dataset."""
        with open(dataset_path, "rb") as handle:
            trajectory_pairs: list[
                tuple[RewardlessTrajectory, RewardlessTrajectory]
            ] = pickle.load(handle)

        steps = [
            (
                (
                    trajectory1[index].astype("float32"),
                    trajectory2[index].astype("float32"),
                )
                for index in range(len(trajectory1))
            )
            for (trajectory1, trajectory2) in trajectory_pairs
        ]

        self.steps = list(chain.from_iterable(steps))

    def __len__(self):
        """Return size of dataset."""
        return len(self.steps)

    def __getitem__(self, index: int):
        """Return item with given index."""
        return self.steps[index][0], self.steps[index][1]


class MultiStepPreferenceDataset(Dataset):
    """PyTorch Dataset for loading preference data for multiple steps."""

    # Specify -1 for `sequence_length` to use the full trajectories
    def __init__(self, dataset_path: str, sequence_length: int):
        """Initialize dataset."""
        with open(dataset_path, "rb") as handle:
            trajectory_pairs: list[
                tuple[RewardlessTrajectory, RewardlessTrajectory]
            ] = pickle.load(handle)

        sequence_pairs: list[tuple[FloatNDArray, FloatNDArray]] = []

        for trajectory1, trajectory2 in trajectory_pairs:
            trajectory_length = (
                len(trajectory1) if sequence_length == -1 else sequence_length
            )

            for index in range(len(trajectory1) - trajectory_length + 1):
                # find the end of this pattern
                end_index = index + trajectory_length

                # gather input and output parts of the pattern
                sequence_pair = (
                    numpy.array(
                        trajectory1[index:end_index],
                        dtype=numpy.float32,
                    ),
                    numpy.array(
                        trajectory2[index:end_index],
                        dtype=numpy.float32,
                    ),
                )
                sequence_pairs.append(sequence_pair)

        self.sequence_pairs = sequence_pairs

    def __len__(self):
        """Return size of dataset."""
        return len(self.sequence_pairs)

    def __getitem__(self, index: int):
        """Return items with given index."""
        return self.sequence_pairs[index]
