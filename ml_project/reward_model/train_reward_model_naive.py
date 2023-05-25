"""Module for training a reward model from trajectory data."""
import pickle
import random
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from network import Network

# Define file paths:
current_path = Path(__file__).parent.resolve()
folder_path = path.join(current_path, "../rl/reward_data")
file_name = path.join(folder_path, "ppo_HalfCheetah-v4_obs_reward_dataset.pkl")

# Load data from file:
with open(file_name, "rb") as handle:
    trajectories = pickle.load(handle)


def sample_preference_batch(trajectories, batch_size):
    """
    Creates a random batch consisting of tuples of observations from two trajectories.
    The tuples are sorted based on the (undiscounted) cumulative reward of the trajectories:
    (obs0, obs1) iff reward1 > reward0.
    """
    batch = []
    for _ in range(batch_size):
        indices = random.sample(list(trajectories.keys()), 2)

        reward0 = np.sum([entry["reward"] for entry in trajectories[indices[0]]])
        obs0 = [entry["obs"] for entry in trajectories[indices[0]]]

        reward1 = np.sum([entry["reward"] for entry in trajectories[indices[1]]])
        obs1 = [entry["obs"] for entry in trajectories[indices[1]]]

        if reward1 > reward0:
            batch.append((obs0, obs1))
        else:
            batch.append((obs1, obs0))

    return batch


# train model using loss in https://arxiv.org/pdf/1904.06387.pdf


def train_reward_model(reward_model, trajectories, epochs, batch_size):
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    # Start training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(trajectories), batch_size):
            # this loop is a bit arbitrarily defined. Here I sample 25 times (length of trajectories) randomly and order
            # tuple of trajectories with the given batch size. I do this for 1 training epoch and hence over 100 epochs.

            # Prepare the inputs for the model
            batch = sample_preference_batch(trajectories, batch_size)
            # The size of the batch corresponds to the number of randomly sampled tuples of trajectories

            probs_sotmax = []
            for traj0, traj1 in batch:
                # For each tuple of trajectories in the batch I create the softmax value.
                # I do this for each trajectory tuple in the batch. Summing those up, I get my final loss.

                rewards0 = []
                rewards1 = []

                for step0, step1 in zip(traj0, traj1):
                    # for each step of the trajectory I predict the reward given the 17 observations

                    # Forward pass - The model outputs the predicted reward for each step
                    reward0 = reward_model(
                        torch.tensor(step0, dtype=torch.float).unsqueeze(0).to(device)
                    )
                    reward1 = reward_model(
                        torch.tensor(step1, dtype=torch.float).unsqueeze(0).to(device)
                    )

                    # Append the predicted rewards
                    rewards0.append(reward0)
                    rewards1.append(reward1)

                # Compute the cumulative rewards
                sum_rewards0 = torch.sum(torch.stack(rewards0))
                sum_rewards1 = torch.sum(torch.stack(rewards1))

                # Compute the probabilities
                prob0 = torch.exp(sum_rewards0) / (
                    torch.exp(sum_rewards0) + torch.exp(sum_rewards1)
                )
                probs_sotmax.append(prob0)

            # Convert probs_sotmax to a tensor
            probs_sotmax = torch.stack(probs_sotmax)

            # Compute the loss over data in batch
            loss = -torch.mean(torch.log(probs_sotmax))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (len(trajectories) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")


# Sample and analyze data:
batch = sample_preference_batch(trajectories, batch_size=2)
input_dim = np.array(batch[0][0][0]).shape[0]

# Initialize network
reward_model = Network(layer_num=3, input_dim=input_dim, hidden_dim=256, output_dim=1)
train_reward_model(reward_model, trajectories, epochs=100, batch_size=2)
