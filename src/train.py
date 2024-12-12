import torch

from models.cnn import TetrisCNN
from models.dqn import DQNAgent
from tetris import Tetris

# initialize env
env = Tetris(20, 10)  # 20 rows, 10 columns

# initialize cnn and dqn
cnn = TetrisCNN()
actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"] 
agent = DQNAgent(cnn, actions, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01)

# train
DQNAgent.train(env, agent, num_episodes=2000)  # Train for 500 episodes

# save
torch.save(cnn.state_dict(), "trained_models/tetris_cnn3.pth")
