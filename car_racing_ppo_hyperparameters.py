import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import cv2

# Hyperparameters
learning_rate = 0.0003
gamma = 0.99
eps_clip = 0.2
K_epochs = 4
num_episodes = 1000  # Define the number of episodes for training

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # For continuous action space, use Tanh activation
        )
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def preprocess_state(self, state):
        # Convert the state to a PyTorch tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return state_tensor

    def select_action(self, state):
        state = state.reshape(-1)  # Flatten the state
        state = self.preprocess_state(state)
        action_mean = self.policy(state)
        m = Categorical(action_mean)
        action = m.sample()
        return action.item(), action_mean

    def update(self, states, actions, rewards, log_probs, advantages):
        old_probs = torch.exp(log_probs.detach())
        for _ in range(K_epochs):
            action_mean = self.policy(states)
            m = Categorical(action_mean)
            entropy = m.entropy().mean()

            new_log_probs = m.log_prob(actions)
            ratios = (new_log_probs - log_probs.detach()).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = (rewards - self.critic(states)).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # Add entropy term to encourage exploration

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    env = gym.make('CarRacing-v2')
    # Get the shape of the observation space
    state_dim = env.observation_space.shape[0]  # Assuming a single observation
    action_dim = env.action_space.shape[0]  # Correctly define the action space dimensionality
    agent = PPO(state_dim, action_dim)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Check if the state is a tuple and extract the first element (assuming it's the image)
            if isinstance(state, tuple):
                state = state[0]
            action, action_mean = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store transitions
            # Update agent using PPO

            state = next_state

        # Plot results (you can use matplotlib or any other plotting library)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")



if __name__ == '__main__':
    main()
