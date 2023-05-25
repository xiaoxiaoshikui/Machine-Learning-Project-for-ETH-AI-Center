import pickle
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from network import Network
from torch.utils.data import DataLoader, Dataset


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


def train_reward_model(reward_model, dataloader, device, epochs):
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
    for epoch in range(epochs):
        epoch_loss = 0
        for traj0, traj1 in dataloader:
            traj0, traj1 = torch.stack(traj0).to(device), torch.stack(traj1).to(device)
            rewards0 = reward_model(traj0)
            rewards1 = reward_model(traj1)
            sum_rewards0 = torch.sum(rewards0)
            sum_rewards1 = torch.sum(rewards1)
            sum_exp_rewards0 = torch.exp(sum_rewards0)
            prob0 = sum_exp_rewards0 / (sum_exp_rewards0 + torch.exp(sum_rewards1))
            loss = -torch.mean(torch.log(prob0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths
    current_path = Path(__file__).parent.resolve()
    folder_path = path.join(current_path, "../rl/reward_data")
    file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")

    # Initialize Dataset and Dataloader
    dataset = TrajectoryDataset(file_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Sample and analyze data
    input_dim = dataset[0][0][0].shape[0]

    # Initialize network
    reward_model = Network(
        layer_num=3, input_dim=input_dim, hidden_dim=256, output_dim=1
    ).to(device)

    # Train model
    train_reward_model(reward_model, dataloader, device, epochs=100)


if __name__ == "__main__":
    main()
