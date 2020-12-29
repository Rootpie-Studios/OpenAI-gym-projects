import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

from DQN import DQN

import signal
import sys
import keras

def signal_handler(sig, frame):
    print("trying to save model")
    global agent
    global save
    agent.model.save(save)
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

save = input("Please enter name to save: ")

agent = DQN(epsilon=0.99, gamma=0.99, batch_size=128, epsilon_min=0.1, lr=0.001, epsilon_decay=0.95, epochs=5)

env = gym.make("LunarLander-v2")

state = env.reset()
state = np.reshape(state, [1, 8])

# Explore 100% 1000 games
print("Exploring 1000 games")
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

# Train with Exploration decline to 1% 300 games
print("Exploring 100 games with declaining epsilon")
for ep in range(1000):
    if ep % 10 == 0:
        agent.update_epsilon()
    tot_reward = 0

    done = False
    # Moves in game
    while not done:

        if agent.epsilon > np.random.rand():
            # Make a random action
            action = env.action_space.sample()
        else:
            # Get action from Q-network
            Qs = agent.model(state, training=False)[0]
            action = np.argmax(Qs)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 8])
        tot_reward += reward

        if done:
            next_state = np.zeros(state.shape)

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(tot_reward),
                  'Explore P: {:.4f}'.format(agent.epsilon))

            agent.memory.append((state, action, reward, next_state))

            state = env.reset()
            state = np.reshape(state, [1, 8])
            
        else:
            agent.memory.append((state, action, reward, next_state))
            state = next_state

    if (ep+1) % 10 == 0:
        agent.train_on_minibatch(times=10)

env.close()

print("Saving model")
agent.model.save(save)



