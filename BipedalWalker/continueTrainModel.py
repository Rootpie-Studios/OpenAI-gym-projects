import gym
import signal
import sys
import keras
import random
import numpy as np
from operator import itemgetter

from GenModel import GenModel
import DiscreteActions

def signal_handler(sig, frame):
    print("trying to save model")
    global agent
    global save
    agent.model.save(save)
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

action_space = DiscreteActions.create_actions()

name = input("Please enter model to load: ")
save = input("Please enter name to save: ")

model = keras.models.load_model(name)

agent = GenModel(model=model, action_space=len(action_space), epsilon=0.2, gamma=0.99, epsilon_min=0.01, lr=0.1, epsilon_decay=0.999, epochs=1)

env = gym.make("BipedalWalker-v3")

# memory = []
state = env.reset()
state = np.array(state)

# Train with Exploration decline to 1% 300 games
print("Exploring 10000 games with declaining epsilon")
for ep in range(10000):
    tot_reward = 0

    states = []
    actions = []
    # rewards = []

    done = False
    # Moves in game
    while not done:

        if agent.epsilon > np.random.rand():
            # Make a random action
            # action = env.action_space.sample()
            action_index = random.randint(0, random.randrange(len(action_space)))
            action = np.array(action_space[action_index])
        else:
            # Get action from Q-network
            Qs = agent.model(np.reshape(state, [1, agent.state_space]), training=False)[0]
            action_index = np.argmax(Qs)
            action = action_space[action_index]

        next_state, reward, done, info = env.step(action)
        tot_reward += reward

        # agent.train_step(state, action, reward, 0.25)

        if done:

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(tot_reward),
                  'Explore P: {:.4f}'.format(agent.epsilon))

            states.append(state)
            actions.append(action_index)
            # rewards.append(reward)

            state = env.reset()
            state = np.array(state)

            agent.train_game(states, actions, tot_reward)
            agent.update_epsilon()
            # memory.append((states, actions, rewards))
            
        else:
            states.append(state)
            actions.append(action_index)
            # rewards.append(reward)

            state = next_state
            state = np.array(state)

    if (ep + 1) % 10 == 0:
        try:
            print("Saving model")
            agent.model.save(save)   
        except:
            print("Save failed")

env.close()

print("Saving model")
agent.model.save(save)