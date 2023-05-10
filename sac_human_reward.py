
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Define the neural networks for the value, Q, and policy functions
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, num_layers):
        super().__init__()
        self.obs_layers = nn.ModuleList()
        self.obs_layers.append(nn.Linear(obs_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.obs_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.act_layers = nn.ModuleList()
        self.act_layers.append(nn.Linear(act_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.act_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = obs
        for layer in self.obs_layers:
            x = torch.relu(layer(x))
        y = act
        for layer in self.act_layers:
            y = torch.relu(layer(y))
        return self.q(torch.cat([x, y], dim=1))

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action = torch.tanh(mean + std * normal)
        log_prob = torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
        return action, log_prob

# Define the soft update function for updating the target networks
def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = capacity
        self.obs_buf = torch.zeros((capacity, obs_dim))
        self.act_buf = torch.zeros((capacity, act_dim))
        self.rew_buf = torch.zeros(capacity)
        self.next_obs_buf = torch.zeros((capacity, obs_dim))
        self.done_buf = torch.zeros(capacity)
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
            None,
        )



class SAC:
    def __init__(self, obs_dim, act_dim, hidden_dim, num_layers, replay_buffer, target_update_freq):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        
        # Define the networks
        self.q1_net = QNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.q2_net = QNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.v_net = ValueNetwork(obs_dim, hidden_dim, num_layers)
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        
        # Define the optimizers
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=1e-3)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=1e-3)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
    
    def update(self, batch_size):
        obs, act, rew, next_obs, done, _ = self.replay_buffer.sample(batch_size)
        
        # Update Q1
        q1_pred = self.q1_net(obs, act)
        v_next = self.v_net(next_obs)
        q1_target = rew + (1 - done) * v_next
        q1_loss = nn.functional.mse_loss(q1_pred, q1_target.detach())
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Update Q2
        q2_pred = self.q2_net(obs, act)
        q2_target = rew + (1 - done) * v_next
        q2_loss = nn.functional.mse_loss(q2_pred, q2_target.detach())
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update V
        v_pred = self.v_net(obs)
        q1_pred = self.q1_net(obs, self.policy_net(obs)[0])
        q2_pred = self.q2_net(obs, self.policy_net(obs)[0])
        q_min = torch.min(q1_pred, q2_pred)
        v_target = q_min - self.policy_net(obs)[1]
        v_loss = nn.functional.mse_loss(v_pred, v_target.detach())
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # Update policy
        q1_pred = self.q1_net(obs, self.policy_net(obs)[0])
        q2_pred = self.q2_net(obs, self.policy_net(obs)[0])
        q_min = torch.min(q1_pred, q2_pred)
        policy_loss = (self.policy_net(obs)[1] - q_min).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    
    def update_target_networks(self):
        soft_update(self.v_target_net, self.v_net, self.tau)
        soft_update(self.q1_target_net, self.q1_net, self.tau)
        soft_update(self.q2_target_net, self.q2_net, self.tau)

    # the training part for the SAC algorithm
    def train(self, env_name, num_epochs, num_steps_per_epoch, batch_size, max_episode_length):
        env = gym.make(env_name)
        obs = env.reset()
        epoch = 0
        
        for t in range(num_epochs * num_steps_per_epoch):
            if t < self.replay_buffer.capacity:
                # Randomly sample from the buffer until it's full
                pass
            else:
                # Sample a batch from the buffer and update the networks
                self.update(batch_size)
                if t % target_update_freq == 0:
                    self.update_target_networks()
                
            # Run one episode and add the transitions to the buffer
            episode_reward = 0
            episode_length = 0
            while True:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act_tensor, _ = self.policy_net(obs_tensor)
                    act = act_tensor.numpy()[0]
                next_obs, rew, done, _ = env.step(act)
                
                self.replay_buffer.add(obs, act, rew, next_obs, done)
                obs = next_obs
                episode_reward += rew
                episode_length += 1
                
                if done or episode_length >= max_episode_length:
                    obs = env.reset()
                    epoch += 1
                    break
                
            if t % num_steps_per_epoch == 0:
                print(f"Epoch: {epoch}, Reward: {episode_reward}, Episode Length: {episode_length}")
