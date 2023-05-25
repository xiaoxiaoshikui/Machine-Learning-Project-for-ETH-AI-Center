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


# somehow validation loss does not seem to work properly. How could I improve on that? Maybe I use to little data?


def train_reward_model(
    reward_model, trajectories, epochs, batch_size, split_ratio=0.8, patience=10
):
    """Train a reward model given trajectories data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    # Partition the data according to batch size
    partitions = partition_data(trajectories, batch_size)
    # Split into training and validation sets
    train_size = int(len(partitions) * split_ratio)
    train_partitions = partitions[:train_size]
    val_partitions = partitions[train_size:]

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    best_model_state = reward_model.state_dict()  # Initialize with the initial state
    for epoch in range(epochs):
        # Shuffle the partitions at the start of each epoch
        random.shuffle(train_partitions)

        # Training
        train_loss = 0
        for pairs in train_partitions:
            batch = generate_batch(trajectories, pairs)
            probs_softmax, loss = compute_loss(batch, reward_model, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_partitions)

        # Validation
        with torch.no_grad():
            val_loss = 0
            for pairs in val_partitions:
                batch = generate_batch(trajectories, pairs)
                _, loss = compute_loss(batch, reward_model, device)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_partitions)

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


def compute_loss(batch, model, device):
    """
    Computes the loss for a batch of data.
    """
    traj0_batch = torch.stack([traj0.clone().detach() for traj0, _ in batch]).to(device)
    traj1_batch = torch.stack([traj1.clone().detach() for _, traj1 in batch]).to(device)

    rewards0 = model(traj0_batch).sum(dim=1)
    rewards1 = model(traj1_batch).sum(dim=1)

    probs_softmax = torch.exp(rewards0) / (torch.exp(rewards0) + torch.exp(rewards1))
    loss = -torch.sum(torch.log(probs_softmax))
    return probs_softmax, loss


def main():
    # File paths
    current_path = Path(__file__).parent.resolve()
    folder_path = path.join(current_path, "../rl/reward_data")
    file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")

    # Load data
    trajectories = load_data(file_name)

    # # Sample and analyze data
    # batch = generate_batch(trajectories, partition_data(trajectories, batch_size=1)[0])
    # input_dim = np.array(batch[0][0][0]).shape[0]

    # Initialize network
    reward_model = Network(layer_num=3, input_dim=17, hidden_dim=256, output_dim=1)

    # Train model
    train_reward_model(reward_model, trajectories, epochs=100, batch_size=1)

    # That's how we can call the best model parameters for later use
    # model = Network(...)  # create a new instance of your model
    # model.load_state_dict(torch.load('best_model_state.pth'))  # load the state dict from file


if __name__ == "__main__":
    main()
