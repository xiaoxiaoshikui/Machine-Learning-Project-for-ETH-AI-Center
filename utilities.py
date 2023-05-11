import matplotlib.pyplot as plt

def visualize_training_progress(rewards, episode_lengths):
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    
    # Plot episode lengths
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
