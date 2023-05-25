import pickle
import random
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from network import Network


def load_data(file_path):
    """Load trajectory data from pickle file."""
    with open(file_path, "rb") as handle:
        trajectories = pickle.load(handle)
    return trajectories


def partition_data(trajectories, batch_size):
    """
    Partitions the trajectories into non-overlapping batches of pairs.
    """
    random.seed(123)  # set the seed
    indices = list(trajectories.keys())
    random.shuffle(indices)  # shuffle the indices to introduce randomness

    # Pair indices
    pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]
    # Handle case of an odd number of indices by pairing last one with a random one
    if len(indices) % 2 == 1:
        pairs.append((indices[-1], random.choice(indices[:-1])))

    # Partition pairs into batches
    partitions = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

    return partitions


def generate_batch(trajectories, pairs):
    """
    Generates a batch using the provided pairs of indices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = []
    for pair in pairs:
        obs0 = torch.tensor(
            np.array([entry["obs"] for entry in trajectories[pair[0]]]),
            dtype=torch.float,
        ).to(device)
        obs1 = torch.tensor(
            np.array([entry["obs"] for entry in trajectories[pair[1]]]),
            dtype=torch.float,
        ).to(device)

        reward0 = np.sum([entry["reward"] for entry in trajectories[pair[0]]])
        reward1 = np.sum([entry["reward"] for entry in trajectories[pair[1]]])

        if reward1 > reward0:
            batch.append((obs0, obs1))
        else:
            batch.append((obs1, obs0))

    return batch


def compute_loss(batch, model, device):
    """
    Computes the loss for a batch of data.
    """
    traj0_batch = torch.stack([traj0.clone().detach() for traj0, _ in batch]).to(device)
    traj1_batch = torch.stack([traj1.clone().detach() for _, traj1 in batch]).to(device)

    rewards0 = model(traj0_batch).sum(dim=1)
    rewards1 = model(traj1_batch).sum(dim=1)

    probs_softmax = torch.exp(rewards0) / (torch.exp(rewards0) + torch.exp(rewards1))
    loss = -torch.mean(torch.log(probs_softmax))
    return probs_softmax, loss


def train_reward_model(reward_model, trajectories, epochs, batch_size):
    """Train a reward model given trajectories data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    partitions = partition_data(trajectories, batch_size)

    for epoch in range(epochs):
        random.shuffle(partitions)  # shuffle partitions at the start of each epoch
        epoch_loss = 0
        for pairs in partitions:
            batch = generate_batch(trajectories, pairs)
            _, loss = compute_loss(batch, reward_model, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(partitions)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")


def main():
    # File paths
    current_path = Path(__file__).parent.resolve()
    folder_path = path.join(current_path, "../rl/reward_data")
    file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")

    # Load data
    trajectories = load_data(file_name)

    # Sample and analyze data
    # batch = generate_batch(trajectories, partition_data(trajectories, batch_size=1)[0])
    # input_dim = np.array(batch[0][0][0]).shape[0]

    # Initialize network
    reward_model = Network(layer_num=3, input_dim=17, hidden_dim=256, output_dim=1)

    # Train model
    train_reward_model(reward_model, trajectories, epochs=100, batch_size=1)


if __name__ == "__main__":
    main()
