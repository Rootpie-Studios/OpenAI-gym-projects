import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space=4, state_space=8, model=False, epsilon=0.9, gamma=0.99, batch_size=16, epsilon_min=0.01, lr=0.001, epsilon_decay=0.997, epochs=1):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.epochs = epochs
        self.memory = deque(maxlen=1000000)
        if model:
            print("Applying previous model")
            self.model = model
        else:
            self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation="relu"))
        model.add(Dense(150, activation="relu"))
        model.add(Dense(150, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

    def sample(self):
        idx = np.random.choice(np.arange(len(self.memory)),
                size=self.batch_size,
                replace=False)
        return [self.memory[ii] for ii in idx]

    def train_on_minibatch(self, times):
        for time in range(times):
            print(str(round((time+1)/times * 100, 2)) + "% Done", end="\r", flush=True)
            inputs = np.zeros((self.batch_size, self.state_space))
            targets = np.zeros((self.batch_size, self.action_space))

            minibatch = self.sample()
            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                inputs[i:i+1] = state_b
                target = reward_b
                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    target_Q = self.model(next_state_b, training=False)[0]
                    target = reward_b + self.gamma * np.amax(self.model(next_state_b, training=False)[0])
                targets[i] = self.model(state_b, training=False)
                targets[i][action_b] = target

            self.model.fit(inputs, targets, epochs=self.epochs, verbose=0)
