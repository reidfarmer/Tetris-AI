import numpy as np
import pygame
import torch
import torch.nn as nn
import random
class DQNAgent:
    def __init__(self, cnn, actions, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.cnn = cnn
        self.actions = actions
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = []
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            q_values = self.cnn(state)
            return self.actions[torch.argmax(q_values).item()]  # Exploitation

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.cnn(states).gather(1, actions).squeeze(1)
        next_q_values = self.cnn(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(env, agent, num_episodes=500, visualize=True):
        for episode in range(num_episodes):
            env.reset()  # Reset the environment for a new game
            state = env.get_board_state()
            done = False
            total_reward = 0

            # Visualization setup
            if visualize:
                pygame.init()
                size = (400, 600)
                screen = pygame.display.set_mode(size)
                pygame.display.set_caption("Tetris Training")

            while not done:
                if visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                    screen.fill((230, 230, 230))
                    env.draw_grid(screen)
                    env.draw_piece(screen)
                    pygame.display.flip()

                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.append((state, agent.actions.index(action), reward, next_state, done))
                state = next_state
                total_reward += reward

                agent.train_step()

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Final Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

            if visualize:
                pygame.quit()

