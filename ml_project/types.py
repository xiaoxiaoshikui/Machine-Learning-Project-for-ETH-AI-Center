"""Common types for all scripts in the project."""

from typing import TypedDict


class Step(TypedDict):
    """Trajectory dictionary type."""

    obs: object
    reward: float


Trajectories = dict[int, list[Step]]
