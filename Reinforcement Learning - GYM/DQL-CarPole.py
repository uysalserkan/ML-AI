# %% Import Section

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# %% Define DQL Class


class DQLAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.gamma = 0.95
        self.lr = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001

        self.memory = deque(maxlen=1000)
        self.model = self.generate_model()

    def generate_model(self):
        model = Sequential([
            Dense(units=48, input_dim=self.state_size, activation="tanh"),
            Dense(units=self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        else:
            minibatch = random.sample(self.memory, batch_size)

            for state, action, reward, next_state, done in minibatch:
                if done:
                    target = reward

                else:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                train_target = self.model.predict(state)
                train_target[0][action] = target

                self.model.fit(state, train_target, verbose=0)

    def adaptiveegreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():

    EPISODES = 10
    BATCH_SIZE = 16
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env=env)
    for episode in range(EPISODES):
        # ...
        state = env.reset()
        state = np.reshape(state, [1, 4])

        time = 0
        while True:

            action = agent.act(state=state, env=env)
            new_state, reward, done, _ = env.step(action=action)
            new_state = np.reshape(new_state, [1, 4])

            agent.remember(state=state, action=action, reward=reward, next_state=new_state, done=done)
            state = new_state
            agent.replay(batch_size=BATCH_SIZE)

            agent.adaptiveegreedy()

            time += 1
            if done:
                print(f"Episode: {episode}\tTime: {time}")
                break

if __name__ == '__main__':
    main()

# %% Test Section
# Bu kısım çalışmayacaktır main fonksiyonu oluşturduğumuz için.
# Eğer görmek istiyorsanız main fonksiyonunu kaldırıp standart tanımla ile devam edebilrisiniz.

trained_model = agent
state = env.reset()
state = np.reshape(state, [1, 4])
time_total = 0
while True:
    env.render()
    act = trained_model.act(state=state)
    next_state, reward, done, _ = env.step(act)
    next_step = np.reshape(next_state, [1, 4])
    state = next_state
    time_total += 1
    if done:
        break
