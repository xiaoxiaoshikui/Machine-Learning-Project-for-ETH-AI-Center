"""Module for creating a dataset from preference data and corresponding observations."""
import pickle
import re
from os import path
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

current_path = Path(__file__).parent.resolve()
preference_data_path = path.join(current_path, "..", "output", "preferences.csv")
observation_data_path = path.join(
    current_path,
    "..",
    "rl",
    "observation_data_corresponding_to_videos",
    "ppo_HalfCheetah-v3_obs_dataset.pkl",
)


def extract_number(input_text: str):
    """Extract step number to get the observations corresponding to the video."""
    text = re.search(r"[0-9]+\.", input_text)
    assert text is not None
    text = text.group()

    num = int(re.sub(r"\D", "", text))

    return num


def load_data(file_path):
    """Load data from pickle file."""
    with open(file_path, "rb") as handle:
        data = pickle.load(handle)
    return data


def save_data(file_path, data):
    """Save data to pickle file."""
    with open(
        file_path,
        "wb",
    ) as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_sorted_indices(preferences_raw: NDArray):
    """Get tuples of indices of videos/trajectories sorted by preference."""
    indices = []
    for line in preferences_raw:
        left_idx = extract_number(line[1])
        right_idx = extract_number(line[2])

        preference = line[3]

        match preference:
            case "left":
                indices.append((right_idx, left_idx))
            case "right":
                indices.append((left_idx, right_idx))
            case "equal":
                indices.append((left_idx, right_idx))
                indices.append((right_idx, left_idx))
            case _:
                raise ValueError(
                    "Preference must be one of these: 'left', 'right', 'equal'"
                )

    return indices


def main():
    """Run preference creation."""
    preferences_raw = np.loadtxt(
        preference_data_path, dtype=str, delimiter=";", skiprows=1
    )

    preference_indices = get_sorted_indices(preferences_raw)

    observations = load_data(observation_data_path)

    preference_dataset = [
        (observations[a], observations[b]) for a, b in preference_indices
    ]

    save_data(path.join(current_path, "preference_dataset.pkl"), preference_dataset)


if __name__ == "__main__":
    main()
