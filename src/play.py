import pygame
import torch
from models.cnn import TetrisCNN
from models.dqn import DQNAgent
from tetris import Tetris

# Step 1: Load the trained CNN model
cnn = TetrisCNN()
cnn.load_state_dict(torch.load("tetris_cnn.pth"))  # Load saved weights
cnn.eval()  # Set to evaluation mode

# Step 2: Initialize the Tetris environment
env = Tetris(20, 10)
actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"]

# Step 3: Initialize the agent
agent = DQNAgent(cnn, actions, epsilon=0)  # Use greedy policy for testing

# Step 4: Let the model play the game
pygame.init()
size = (400, 600)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("RL Tetris")
clock = pygame.time.Clock()
fps = 30

print("Testing the trained model...")
state = env.get_board_state()
total_reward = 0
running = True

while running:
    screen.fill((230, 230, 230))

    # RL Agent selects an action
    # RL Agent selects an action
    action = agent.select_action(state)
    print(f"Selected Action: {action}")  # Debug log
    next_state, reward, done = env.step(action)
    print(f"Reward: {reward}, Done: {done}")  # Debug log

    state = next_state
    total_reward += reward

    # Render the game
    env.draw_grid(screen)
    env.draw_piece(screen)
    env.display_stats(screen)

    if env.state == "gameover" or done:
        env.display_game_over(screen)
        print(f"Game Over! Total Reward: {total_reward}")
        running = False

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
