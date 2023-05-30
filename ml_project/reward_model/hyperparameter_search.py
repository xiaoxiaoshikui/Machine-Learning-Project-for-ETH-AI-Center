"""Module for seraching for hyperparameters for a finetuning model and its training."""
import os
from os import path
from pathlib import Path

import ray
import torch
from ray import tune
from ray.tune import TuneConfig, Tuner
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from .datasets import MultiStepPreferenceDataset
from .finetune_reward_model import train_reward_model
from .networks import LightningTrajectoryNetwork

SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    # "batch_size": tune.randint(1, 100),
    "sequence_length": tune.randint(1, 100),
    "hidden_dim": tune.lograndint(1, 1000),
    "layer_num": tune.lograndint(1, 1000),
}

script_path = Path(__file__).parent.resolve()
dataset_path = path.join(script_path, "preference_dataset.pkl")


def train(config: dict[str, int]):
    """Train the network using the selected hyperparameters."""
    dataset = MultiStepPreferenceDataset(
        dataset_path, sequence_length=config["sequence_length"]
    )

    reward_model = LightningTrajectoryNetwork(
        input_dim=17,
        hidden_dim=config["hidden_dim"],
        layer_num=config["layer_num"],
        output_dim=1,
        learning_rate=config["learning_rate"],
    )

    train_reward_model(
        reward_model,
        dataset,
        epochs=1000,
        batch_size=1,
        enable_progress_bar=False,
        callback=TuneReportCallback({"val_loss": "val_loss"}, on="validation_end"),
    )


def main():
    """Run hyperparameter search."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    gpu_count = torch.cuda.device_count()

    ray.init(num_cpus=cpu_count, num_gpus=gpu_count)

    worker_count = 5

    trainable = tune.with_resources(
        train, {"cpu": cpu_count // worker_count, "gpu": gpu_count / worker_count}
    )
    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=100,
            reuse_actors=False,
            scheduler=ASHAScheduler(metric="val_loss", mode="min"),
            search_alg=OptunaSearch(
                metric="val_loss",
                mode="min",
                points_to_evaluate=[
                    {
                        "learning_rate": 0.000397989,
                        "sequence_length": 32,
                        "hidden_dim": 984,
                        "layer_num": 7,
                    },
                    {
                        "learning_rate": 0.00111152,
                        "sequence_length": 32,
                        "hidden_dim": 777,
                        "layer_num": 33,
                    },
                    {
                        "learning_rate": 0.0031431,
                        "sequence_length": 32,
                        "hidden_dim": 44,
                        "layer_num": 2,
                    },
                ],
                evaluated_rewards=[0.265849, 2.07251, 0.260904],
            ),
        ),
        param_space=SEARCH_SPACE,
    )
    results = tuner.fit()

    print(f"Best config: {results}")


if __name__ == "__main__":
    main()
