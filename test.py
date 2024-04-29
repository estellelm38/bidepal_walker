import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# Définition des ensembles d'hyperparamètres à tester
hyperparams_sets = [
    {'batch_size': 64, 'gamma': 0.99, 'learning_rate': 0.00025},
    {'batch_size': 128, 'gamma': 0.99, 'learning_rate': 0.0005},
    {'batch_size': 128, 'gamma': 0.99, 'learning_rate': 0.0005},
    {'batch_size': 128, 'gamma': 0.99, 'learning_rate': 0.0005},
    {'batch_size': 128, 'gamma': 0.99, 'learning_rate': 0.0005},
    {'batch_size': 128, 'gamma': 0.99, 'learning_rate': 0.0005}
]

# Création de l'environnement
env_name = 'CarRacing-v2'
env = DummyVecEnv([lambda: gym.make(env_name)])

# Nombre de fois pour répéter l'entraînement et le test pour chaque ensemble d'hyperparamètres
num_runs = 5

# Stockage des récompenses moyennes pour chaque ensemble d'hyperparamètres
all_avg_ep_rewards = []

# Boucle sur chaque ensemble d'hyperparamètres
for hyperparams in hyperparams_sets:
    avg_ep_rewards = []
    for _ in range(num_runs):
        # Initialisation du modèle avec les hyperparamètres spécifiés
        model = PPO(MlpPolicy, env, verbose=1, **hyperparams)

        # Entraînement du modèle
        model.learn(total_timesteps=10000)

        # Test du modèle
        number_of_episodes = 3
        ep_rewards = []
        obs = env.reset()
        for _ in range(number_of_episodes):
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            ep_rewards.append(episode_reward)
            obs = env.reset()

        # Calcul de la récompense moyenne par épisode
        avg_ep_reward = np.mean(ep_rewards)
        avg_ep_rewards.append(avg_ep_reward)

    # Enregistrer les récompenses moyennes pour cet ensemble d'hyperparamètres
    all_avg_ep_rewards.append(avg_ep_rewards)

# Tracer les courbes de récompenses moyennes pour chaque ensemble d'hyperparamètres
for i, avg_ep_rewards in enumerate(all_avg_ep_rewards):
    plt.plot(avg_ep_rewards, label=f'Set {i+1}')

plt.xlabel('Run')
plt.ylabel('Récompense moyenne par épisode')
plt.title('Récompenses moyennes par épisode pour différents ensembles d\'hyperparamètres (moyenne sur 5 runs)')
plt.legend()
plt.show()
plt.savefig(f'/home/estelle/robotlearn/internship/{env_name}_rewards_per_episode')
