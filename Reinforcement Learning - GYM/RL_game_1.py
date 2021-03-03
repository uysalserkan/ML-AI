import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

WIDTH = 480
HEIGHT = 480
FPS = 30

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25, 25))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 1
        self.x_speed = 0

    def update(self, action):
        self.x_speed = 0
        keyState = pygame.key.get_pressed()

        if keyState[pygame.K_LEFT] or action == 0:
            self.x_speed = -5
        elif keyState[pygame.K_RIGHT] or action == 1:
            self.x_speed = 5
        else:
            self.x_speed = 0

        self.rect.x += self.x_speed

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

    def getCoordinate(self):
        return (self.rect.x, self.rect.y)

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, BLACK, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(1,6)

        self.speed_x = 0
        self.speed_y = 3

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(1, 6)
            # self.speed_y = 3

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)




class DQLAgent:
    def __init__(self):
        self.state_size = 4 # distance[(p1-e1)x, (p1,e1)y, (p1,e2)x, (p1,e2)y]
        self.action_size = 3 # right, left, no-move

        self.gamma = 0.95
        self.lr = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001

        self.memory = deque(maxlen=1000)
        self.model = self.generate_model()

    def generate_model(self):
        model = Sequential([
            Dense(units=48, input_dim=self.state_size, activation="relu"),
            Dense(units=self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        else:
            minibatch = random.sample(self.memory, batch_size)

            for state, action, reward, next_state, done in minibatch:
                state = np.array(state)
                next_state = np.array(next_state)
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


class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.all_enemy = pygame.sprite.Group()
        self.p1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.all_sprite.add(self.p1)
        self.all_sprite.add(self.e1)
        self.all_sprite.add(self.e2)
        self.all_enemy.add(self.e1)
        self.all_enemy.add(self.e2)

        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()

    @staticmethod
    def getDistance(x, y):
        return x - y

    def step(self, action):
        state_list = []

        self.p1.update(action)
        self.all_enemy.update()

        next_player_state = self.p1.getCoordinate()
        en1 = self.e1.getCoordinates()
        en2 = self.e2.getCoordinates()

        state_list.append(self.getDistance(next_player_state[0], en1[0]))
        state_list.append(self.getDistance(next_player_state[1], en1[1]))
        state_list.append(self.getDistance(next_player_state[0], en2[0]))
        state_list.append(self.getDistance(next_player_state[1], en2[1]))

        return [state_list]

    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.all_enemy = pygame.sprite.Group()
        self.p1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.all_sprite.add(self.p1)
        self.all_sprite.add(self.e1)
        self.all_sprite.add(self.e2)
        self.all_enemy.add(self.e1)
        self.all_enemy.add(self.e2)

        self.reward = 0
        self.total_reward = 0
        self.done = False

        state_list = []

        next_player_state_real = self.p1.getCoordinate()
        en1_real = self.e1.getCoordinates()
        en2_real = self.e2.getCoordinates()

        state_list.append(self.getDistance(next_player_state_real[0], en1_real[0]))
        state_list.append(self.getDistance(next_player_state_real[1], en1_real[1]))
        state_list.append(self.getDistance(next_player_state_real[0], en2_real[0]))
        state_list.append(self.getDistance(next_player_state_real[1], en2_real[1]))

        return [state_list], self.reward

    def run(self):
        state = self.initialStates()
        running = True
        batch_size = 24
        while running:
            self.reward = 2
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward

            hits = pygame.sprite.spritecollide(self.p1, self.all_enemy, False,
                                               pygame.sprite.collide_circle)  # circle alternatifi -> rect
            if hits:
                self.reward = -150
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Çarpışma Gerçekleşti.\nTotal Reward:", self.total_reward)

            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            self.agent.replay(batch_size=batch_size)

            self.agent.adaptiveegreedy()

            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            pygame.display.flip()

        pygame.quit()

# %% Main part


"""all_sprite = pygame.sprite.Group()
all_enemy = pygame.sprite.Group()
p1 = Player()
e1 = Enemy()
e2 = Enemy()
all_sprite.add(p1)
all_sprite.add(e1)
all_sprite.add(e2)
all_enemy.add(e1)
all_enemy.add(e2)"""


if __name__ == "__main__":
    env = Env()
    reward_list = [    ]
    time_t = 0
    while True:
        time_t += 1
        reward_list.append(env.total_reward)

        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("UYSAL Game")
        clock = pygame.time.Clock()

        env.run()
        print("Episode {}, Reward {}".format(time_t, env.total_reward))

"""
running = Truerepla
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        all_sprite.update()
        hits = pygame.sprite.spritecollide(p1, all_enemy, False, pygame.sprite.collide_circle) # circle alternatifi -> rect
        if hits:
            running = False
            print("Çarpışma Gerçekleşti.")

        screen.fill(GREEN)
        all_sprite.draw(screen)
        pygame.display.flip()

pygame.quit()
"""
