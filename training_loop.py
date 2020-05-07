"""
Training loop for the Coma framework on Switch2-v0 the ma_gym
"""

import gym
import ma_gym
import matplotlib.pyplot as plt
import numpy as np

from COMA import COMA


def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


if __name__ == "__main__":
    # Hyperparameters
    agent_num = 2

    state_dim = 2
    action_dim = 5

    gamma = 0.99
    lr_a = 0.0001
    lr_c = 0.005

    target_update_steps = 10

    # agent initialisation

    agents = COMA(agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps)

    env = gym.make("Switch2-v0")
    obs = env.reset()

    episode_reward = 0
    episodes_reward = []

    # training loop

    n_episodes = 10000
    episode = 0

    while episode < n_episodes:
        actions = agents.get_actions(obs)
        next_obs, reward, done_n, _ = env.step(actions)

        agents.memory.reward.append(reward)
        for i in range(agent_num):
            agents.memory.done[i].append(done_n[i])

        episode_reward += sum(reward)

        obs = next_obs

        if all(done_n):
            episodes_reward.append(episode_reward)
            episode_reward = 0

            episode += 1

            obs = env.reset()

            if episode % 10 == 0:
                agents.train()

            if episode % 100 == 0:
                print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
