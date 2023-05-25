"""Common types for all scripts in the project."""

from typing import TypedDict

from torch import Tensor


class Step(TypedDict):
    """Trajectory dictionary type."""

    obs: object
    reward: float


Trajectories = dict[int, list[Step]]
Batch = list[tuple[list[object], list[object]]]
TensorBatch = list[tuple[Tensor, Tensor]]
