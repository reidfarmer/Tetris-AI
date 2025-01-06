from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

class DQNAgent:
    def __init__(
        self, 
        cnn, 
        actions, 
        lr=1e-3, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        min_epsilon=0.01,
        max_replay_buffer_size=20000,
        batch_size=64,
        target_update_interval=1000,
        use_decaying_discount=False,
        discount_start=0.8,
        discount_end=0.94,
        discount_duration=4000
    ):
        """
        :param cnn: The policy neural network (torch.nn.Module)
        :param actions: List of possible actions
        :param lr: Learning rate
        :param gamma: Initial discount factor for Q-learning
        :param epsilon: Initial epsilon for epsilon-greedy
        :param epsilon_decay: Multiplicative decay for epsilon
        :param min_epsilon: Minimum exploration rate
        :param max_replay_buffer_size: Maximum size of the replay buffer
        :param batch_size: Batch size for training
        :param target_update_interval: Number of steps after which we update the target network
        :param use_decaying_discount: Whether to use a decaying discount factor
        :param discount_start: Starting discount factor if decaying discount is used
        :param discount_end: Ending discount factor if decaying discount is used
        :param discount_duration: Number of steps over which discount is decayed
        """
        self.cnn = cnn
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_replay_buffer_size)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.target_cnn = type(cnn)()  # create another instance of the same class
        self.target_cnn.load_state_dict(cnn.state_dict())  # copy weights
        self.target_cnn.eval()

        # For target network updates
        self.target_update_interval = target_update_interval
        self.steps_done = 0

        # Decaying discount
        self.use_decaying_discount = use_decaying_discount
        self.discount_start = discount_start
        self.discount_end = discount_end
        self.discount_duration = discount_duration

    def select_action(self, state):
        """
        Choose an action using epsilon-greedy.
        """
        # Epsilon-greedy threshold
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            q_values = self.cnn(state_t)
            return self.actions[torch.argmax(q_values).item()]  # Exploitation

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Sample a mini-batch from memory and update the Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Convert actions to integer indices
        action_indices = [self.actions.index(a) for a in actions]
        action_indices = torch.tensor(action_indices, dtype=torch.int64).unsqueeze(1)

        # Compute Q(s, a) using current network
        q_values = self.cnn(states).gather(1, action_indices).squeeze(1)

        # Compute Q'(s', a') using target network
        with torch.no_grad():
            # If we want to incorporate a decaying discount factor:
            gamma = self.get_current_gamma() if self.use_decaying_discount else self.gamma

            # Next state Q-values from target network
            next_q_values = self.target_cnn(next_states).max(1)[0]
            # target = r + gamma * max Q'(s', a') (if not done)
            target = rewards + gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Increment steps_done
        self.steps_done += 1

        # Update target network if needed
        if self.steps_done % self.target_update_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        """
        Copy parameters from the policy network to the target network.
        """
        self.target_cnn.load_state_dict(self.cnn.state_dict())

    def get_current_gamma(self):
        """
        If using decaying discount, interpolate between discount_start and discount_end over discount_duration steps.
        """
        fraction = min(1.0, self.steps_done / float(self.discount_duration))
        return self.discount_start + fraction * (self.discount_end - self.discount_start)

    @staticmethod
    def train(env, agent, num_episodes=2000, visualize=False):
        """
        Train the DQNAgent on the Tetris environment.
        """
        import pygame
        for episode in range(num_episodes):
            env.reset()  # Reset the environment
            state = env.get_board_state()
            done = False
            total_reward = 0

            while not done:
                if visualize:
                    pygame.init()
                    size = (400, 600)
                    screen = pygame.display.set_mode(size)
                    pygame.display.set_caption("Tetris Training")

                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                total_reward += reward

                if visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                    screen.fill((230, 230, 230))
                    env.draw_grid(screen)
                    env.draw_piece(screen)
                    pygame.display.flip()

            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Total Reward: {total_reward}, "
                  f"Final Score: {env.score}, "
                  f"Epsilon: {agent.epsilon:.3f}")

        if visualize:
            pygame.quit()