from collections import deque
import random
import flappy_bird_gym
import tensorflow
import numpy as np
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizer_v2.adam import Adam


class DQNAgent:
    def __init__(self):
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = not self.env.action_space.n
        self.jump_probability = 0.9
        self.model = self.createModel()
        self.memory = deque(maxlen=2000)
        self.train_start = 1000

    def createModel(self):
        model = Sequential()
        model.add(Input(self.state_space,))
        model.add(Dense(128, activation="tanh"))
        model.add(Dense(256, activation="tanh"))
        model.add(Dense(512, activation="tanh"))
        model.add(Dense(2, activation="linear"))
        optimizer = Adam(learning_rate=3e-4, decay=1e-5)


        model.compile(optimizer, 'binary_crossentropy')
        model.summary()
        return model



    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return 1 if np.random.random() < self.jump_probability else 0


    def learn(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_space))
        next_state = np.zeros((self.batch_size, self.state_space))
        action, reward, done = [], [], []


        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def decreaseEpsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))