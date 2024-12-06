import torch

from models.cnn import TetrisCNN
from models.dqn import DQNAgent
from tetris import Tetris

# Step 1: Initialize the Tetris environment
env = Tetris(20, 10)  # 20 rows, 10 columns

# Step 2: Initialize the CNN and DQN agent
cnn = TetrisCNN()  # Neural network model
actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"]  # Possible actions
agent = DQNAgent(cnn, actions, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01)

# Step 3: Train the agent
DQNAgent.train(env, agent, num_episodes=2000)  # Train for 500 episodes

# Step 4: Save the trained model
torch.save(cnn.state_dict(), "trained_models/tetris_cnn3.pth")
