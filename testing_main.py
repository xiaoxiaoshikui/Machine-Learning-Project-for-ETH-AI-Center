import gym
from sac_human_reward import SAC, ReplayBuffer

env_name = 'Pendulum-v1'
num_epochs = 100
num_steps_per_epoch = 1000
batch_size = 128
max_episode_length = 1000
replay_buffer_size = 100000
target_update_freq = 1000
# Create the OpenAI Gym environment
env = gym.make(env_name)

# Create an instance of the ReplayBuffer class with the required arguments
replay_buffer = ReplayBuffer(act_dim=env.action_space.shape[0], obs_dim=env.observation_space.shape[0], capacity=replay_buffer_size)

# Create an instance of the SAC class and pass in the observation and action dimensions
sac = SAC(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], hidden_dim=256, num_layers=3, replay_buffer=replay_buffer, target_update_freq = target_update_freq)

# Train the policy on the environment
sac.train(env_name=env_name, num_epochs=num_epochs, num_steps_per_epoch=num_steps_per_epoch, batch_size=batch_size, max_episode_length=max_episode_length)
