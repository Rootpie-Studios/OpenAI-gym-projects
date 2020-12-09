import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

env = gym.make("LunarLander-v2")

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space=4, state_space=8):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .997
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation="relu"))
        model.add(Dense(120, activation="relu"))
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

agent = DQN()
state = env.reset()
state = np.reshape(state, [1, 8])

# Make a bunch of random actions and store the experiences
for ii in range(1000):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 8])

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        agent.memory.append((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(state, [1, 8])
    else:
        # Add experience to memory
        agent.memory.append((state, action, reward, next_state))
        state = next_state

state = env.reset()
state = np.reshape(state, [1, 8])

# Number of games
for ep in range(400):
    tot_reward = 0

    done = False
    # Moves in game
    while not done:
        if ep + 1 % 30 == 0:
            env.render()

        if agent.epsilon > np.random.rand():
            # Make a random action
            action = env.action_space.sample()
        else:
            # Get action from Q-network
            Qs = agent.model.predict(state)[0]
            action = np.argmax(Qs)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 8])
        tot_reward += reward

        if done:
            next_state = np.zeros(state.shape)
            agent.update_epsilon()

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(tot_reward),
                  'Explore P: {:.4f}'.format(agent.epsilon))

            agent.memory.append((state, action, reward, next_state))

            state = env.reset()
            state = np.reshape(state, [1, 8])
            
        else:
            agent.memory.append((state, action, reward, next_state))
            state = next_state


        # Replay
        inputs = np.zeros((agent.batch_size, agent.state_space))
        targets = np.zeros((agent.batch_size, agent.action_space))

        minibatch = agent.sample()
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target_Q = agent.model.predict(next_state_b)[0]
                target = reward_b + agent.gamma * np.amax(agent.model.predict(next_state_b)[0])
            targets[i] = agent.model.predict(state_b)
            targets[i][action_b] = target
        agent.model.fit(inputs, targets, epochs=1, verbose=0)

env.close()