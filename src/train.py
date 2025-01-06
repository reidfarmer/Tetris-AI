import torch

from models.cnn import TetrisCNN
from models.dqn import DQNAgent
from tetris import Tetris

# Step 1: Initialize the Tetris environment
env = Tetris(20, 10)  # 20 rows, 10 columns

# Step 2: Initialize the CNN and DQN agent
cnn = TetrisCNN()  # Neural network model
actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"]  # Possible actions

# Example improved hyperparameters:
learning_rate = 1e-3
gamma = 0.99
initial_epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
max_replay_buffer_size = 20000
batch_size = 64
target_update_interval = 1000
num_episodes = 2000

# Optionally use decaying discount:
use_decaying_discount = True
discount_start = 0.8
discount_end = 0.94
discount_duration = 4000

agent = DQNAgent(
    cnn,
    actions,
    lr=learning_rate,
    gamma=gamma,
    epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon,
    max_replay_buffer_size=max_replay_buffer_size,
    batch_size=batch_size,
    target_update_interval=target_update_interval,
    use_decaying_discount=use_decaying_discount,
    discount_start=discount_start,
    discount_end=discount_end,
    discount_duration=discount_duration
)

# Step 3: Train the agent
DQNAgent.train(env, agent, num_episodes=num_episodes)  # e.g. train for 2000 episodes

# Step 4: Save the trained model
torch.save(cnn.state_dict(), "trained_models/tetris_cnn5.pth")