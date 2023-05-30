"""Module for fine-tuning a reward model using preference data."""
import math
import os
from os import path
from pathlib import Path
from typing import Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split

from .datasets import MultiStepPreferenceDataset, PreferenceDataset
from .networks import LightningTrajectoryNetwork
from .networks_old import LightningRNNNetwork

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")

# File paths
script_path = Path(__file__).parent.resolve()
file_path = path.join(script_path, "preference_dataset.pkl")
pretrained_model_path = path.join(script_path, "models", "reward_model_pretrained.pth")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8


def train_reward_model(
    reward_model: LightningModule,
    dataset: Union[PreferenceDataset, MultiStepPreferenceDataset],
    epochs: int,
    batch_size: int,
    split_ratio: float = 0.8,
    callback: Union[Callback, None] = None,
    enable_progress_bar=True,
):
    """Train a reward model given preference data."""
    train_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[train_size, len(dataset) - train_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count,
    )

    trainer = Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        enable_progress_bar=enable_progress_bar,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            *([callback] if callback is not None else []),
        ],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    return reward_model


def main():
    """Run reward model fine-tuning."""
    # Train MLP
    # dataset = PreferenceDataset(file_path)

    # reward_model = LightningNetwork(
    #     input_dim=17, hidden_dim=256, layer_num=3, output_dim=1
    # )

    # train_reward_model(reward_model, dataset, epochs=100, batch_size=4)

    # Train MLP using full-trajectory loss
    dataset = MultiStepPreferenceDataset(file_path, sequence_length=70)

    reward_model = LightningTrajectoryNetwork(
        input_dim=40, hidden_dim=256, layer_num=12, output_dim=1, learning_rate=2e-4
    )

    # load pre-trained weights
    reward_model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    train_reward_model(reward_model, dataset, epochs=100, batch_size=1)

    # Train RNN
    dataset = MultiStepPreferenceDataset(file_path, sequence_length=70)

    reward_model = LightningRNNNetwork(
        input_size=40, hidden_size=256, num_layers=12, dropout=0.2
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=1)


if __name__ == "__main__":
    main()
