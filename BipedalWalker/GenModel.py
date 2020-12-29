import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import itertools
import random


class GenModel:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space=4, state_space=24, model=False, epsilon=0.9, gamma=0.99, epsilon_min=0.01, lr=0.001, epsilon_decay=0.997, epochs=5, batch_size=256):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=1000000)


        if model:
            print("Applying previous model")
            self.model = model
        else:
            self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(1500, input_dim=self.state_space, activation="relu"))
        model.add(Dense(1500, activation="relu"))
        model.add(Dense(1500, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def sample(self):
        idx = np.random.choice(np.arange(len(self.memory)),
                size=self.batch_size,
                replace=True)
        return [self.memory[ii] for ii in idx]

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

    def train(self, memory):
        for states, actions, rewards in memory:
            index = 0
            for state, action, reward in zip(states, actions, rewards):
                action_reward = np.array(self.model(np.reshape(state, [1, self.state_space]), training=False)[0])
                action_reward[action] += reward * (self.gamma**(len(states) - index))
                index += 1

                self.model.fit(np.reshape(state, [1, self.state_space]), np.reshape(action_reward, [1, self.action_space]), epochs=self.epochs, verbose=0)

    def train_on_minibatch(self, times):
        for time in range(times):
            print(str(round((time+1)/times * 100, 2)) + "% Done", end="\r", flush=True)
            inputs = np.zeros((self.batch_size, self.state_space))
            targets = np.zeros((self.batch_size, self.action_space))

            minibatch = self.sample()
            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                inputs[i] = state_b
                if not (next_state_b == np.zeros(state_b.shape)).all():
                    target = reward_b + self.gamma * np.amax(self.model(np.reshape(next_state_b, [1, self.state_space]), training=False)[0])
                else:
                    target = reward_b
                targets[i] = np.array(self.model(np.reshape(state_b, [1, self.state_space]), training=False)[0])
                targets[i][action_b] = target

            self.model.fit(inputs, targets, epochs=self.epochs, verbose=0)

    def train_game(self, states, actions, tot_reward):
        index = 0
        for state, action in zip(states, actions):
            print(str(round((index+1)/len(states) * 100, 2)) + "% Done", end="\r", flush=True)
            action_reward = np.array(self.model(np.reshape(state, [1, self.state_space]), training=False)[0])
            action_reward[action] += tot_reward * (self.gamma**(len(states) - index)) * 0.001
            index += 1

            self.model.fit(np.reshape(state/np.linalg.norm(state), [1, self.state_space]), np.reshape(action_reward/np.linalg.norm(action_reward), [1, self.action_space]), epochs=self.epochs, verbose=0)
