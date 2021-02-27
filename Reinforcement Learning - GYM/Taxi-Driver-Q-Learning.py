# %% Import statements
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("Taxi-v3").env
episodes = 10000

# %% Q-Table initialization and Hyperparams
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
epsilon = 0.1
gamma = 0.9

reward_list = []
dropout_list = []

# %% Episodes

for i in range(episodes):
    state = env.reset()
    reward_count = 0
    dropout_count = 0

    while True:
        if random.uniform(0, 1) < epsilon:
            act = env.action_space.sample()
        else:
            act = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(act)

        old_value = q_table[state, act]
        next_max = np.max(q_table[next_state])
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[state, act] = next_value
        state = next_state

        if reward == -10:
            dropout_count += 1

        if done:
            break
        else:
            reward_count += reward

    if i % 10 == 0:
        dropout_list.append(dropout_count)
        reward_list.append(reward_count)
        print("Episode: {}\tReward: {}\tWrong dropout: {}".format(i, reward_count, dropout_count))

# %% Plotting
fig, axes = plt.subplots(1, 2, figsize=(9, 6))
axes[0].plot(reward_list)
axes[0].set_xlabel("Episodes")
axes[0].set_ylabel("Reward")
axes[0].grid(True)

axes[1].plot(dropout_list)
axes[1].set_xlabel("Episodes")
axes[1].set_ylabel("Dropout")
axes[1].grid(True)

plt.show()

# %% Q Table

test_state = env.encode(4, 4, 3, 3)
env.s = test_state
env.render()
