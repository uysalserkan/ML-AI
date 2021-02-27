# %% Import and definition section

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("FrozenLake-v0")

q_table = np.zeros([env.observation_space.n, env.action_space.n])
lr = 0.1
epsilon = 0.1
gamma = 0.9
episodes = 1000
reward_list = []

# %% Initialization section
for i in range(episodes):
    state = env.reset()
    reward_count = 0

    while True:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action=action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        next_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)

        q_table[state, action] = next_value

        state = next_state
        reward_count += reward
        if done:
            break

    if i % 10 == 0:
        reward_list.append(reward_count)
        print(
            "Episode: {} \tReward: {}".format(i, reward_count)
        )

# %% Plotting
plt.plot(reward_list)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
