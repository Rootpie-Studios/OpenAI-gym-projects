import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import signal
import sys
import keras

env = gym.make("LunarLander-v2")

state = env.reset()
state = np.reshape(state, [1, 8])

name = input("Please state model to load: ")

model = keras.models.load_model(name)

for ep in range(10):
    done = False
    tot_reward = 0

    while not done:
        env.render()
        Qs = model(state, training=False)[0]
        action = np.argmax(Qs)

        next_state, reward, done, info = env.step(action)

        tot_reward += reward

        next_state = np.reshape(next_state, [1, 8])

        if done:
            next_state = np.zeros(state.shape)

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(tot_reward))


            state = env.reset()
            state = np.reshape(state, [1, 8])
            
        else:
            state = next_state

env.close()