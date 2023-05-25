"""Module for training a reward model from trajectory data."""
import pickle
import random
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from network import Network
from torch.utils.data import random_split, Dataset


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for loading trajectories data."""

    def __init__(self, file_path):
        with open(file_path, "rb") as handle:
            self.trajectories = pickle.load(handle)
        self.keys = list(self.trajectories.keys())

    def __len__(self):
        return len(self.keys) // 2  # since we pair two trajectories

    def __getitem__(self, idx):
        # Pair trajectories
        key0, key1 = self.keys[2 * idx], self.keys[2 * idx + 1]

        # Get observations
        obs0 = [
            torch.tensor(entry["obs"], dtype=torch.float32)
            for entry in self.trajectories[key0]
        ]
        obs1 = [
            torch.tensor(entry["obs"], dtype=torch.float32)
            for entry in self.trajectories[key1]
        ]

        # Get rewards and decide preference
        reward0 = np.sum([entry["reward"] for entry in self.trajectories[key0]])
        reward1 = np.sum([entry["reward"] for entry in self.trajectories[key1]])

        if reward1 > reward0:
            return obs0, obs1
        else:
            return obs1, obs0


def train_reward_model(
    reward_model, train_dataloader, val_dataloader, epochs, patience=10
):
    """Train a reward model given trajectories data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    best_model_state = reward_model.state_dict()  # Initialize with the initial state
    for epoch in range(epochs):
        # Training
        train_loss = 0
        for batch in train_dataloader:
            obs0, obs1 = batch  # unpack the batch
            obs0 = torch.stack(obs0).to(device)  # stack tensors in the batch
            obs1 = torch.stack(obs1).to(device)  # stack tensors in the batch

            probs_softmax, loss = compute_loss(obs0, obs1, reward_model, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataloader:
                obs0, obs1 = batch  # unpack the batch
                obs0 = torch.stack(obs0).to(device)  # stack tensors in the batch
                obs1 = torch.stack(obs1).to(device)  # stack tensors in the batch

                _, loss = compute_loss(obs0, obs1, reward_model, device)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
        )

        # Early stopping
        delta = 0.001  # minimum acceptable improvement
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            best_model_state = (
                reward_model.state_dict()
            )  # save the parameters of the best model
            torch.save(
                best_model_state, "best_reward_model_state.pth"
            )  # save the parameters for later use to disk
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(
                    f"No improvement after for {patience} epochs, therefore stopping training."
                )
                break  # break instead of return, so that the function can return the best model state

    # hold the weights and biases for the best model state during trainn
    reward_model.load_state_dict(best_model_state)

    # how we can call the parameters for later use
    # model = Network(...)  # create a new instance of your model
    # model.load_state_dict(torch.load('best_model_state.pth'))  # load the state dict from file

    return reward_model




def compute_loss(obs0, obs1, model, device):
    """Compute the loss for a batch of data."""
    rewards0 = model(obs0).sum(dim=1)
    rewards1 = model(obs1).sum(dim=1)

    probs_softmax = torch.exp(rewards0) / (torch.exp(rewards0) + torch.exp(rewards1))
    loss = -torch.sum(torch.log(probs_softmax))
    return probs_softmax, loss


def main():
    """Run reward model training."""
    # File paths
    current_path = Path(__file__).parent.resolve()
    folder_path = path.join(current_path, "../rl/reward_data")
    file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")

    # Load data
    dataset = TrajectoryDataset(file_name)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
    val_size = len(dataset) - train_size  # the rest for validation

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize network
    reward_model = Network(layer_num=3, input_dim=17, hidden_dim=256, output_dim=1)

    # Train model
    train_reward_model(reward_model, train_dataloader, val_dataloader, epochs=100)

    # Load the best model parameters
    # model = Network(...)  # create a new instance of your model
    # model.load_state_dict(torch.load('best_reward_model_state.pth'))  # load the state dict from file

if __name__ == "__main__":
    main()
