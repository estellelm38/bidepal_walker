import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import TimeLimit
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# Définition des hyperparamètres
hyperparams = {
    'n_steps': 2048,
    'batch_size': 64,
    'gamma': 0.99,
    'learning_rate': 0.00025,
}

# Création de l'environnement
env_name = 'CarRacing-v2'
env = DummyVecEnv([lambda: gym.make(env_name,
                                    #render_mode="human"
                                    )])

# Initialisation du modèle avec les hyperparamètres spécifiés
model = PPO(MlpPolicy, env, verbose=1, **hyperparams)

# Entraînement du modèle
model.learn(total_timesteps=10000)

number_of_episodes=20

ep_rewards = []
obs = env.reset()
for _ in range(number_of_episodes):
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        #env.render()
    ep_rewards.append(episode_reward)
    obs = env.reset()

# Tracer la courbe de récompenses par épisode
plt.plot(ep_rewards)
plt.xlabel('Episode')
plt.ylabel('Récompense')
plt.title('Récompenses par épisode')
plt.show()
