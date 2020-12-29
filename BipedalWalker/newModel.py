import gym
import signal
import sys
import random
import numpy as np
from operator import itemgetter

from GenModel import GenModel
import DiscreteActions

def signal_handler(sig, frame):
    print("Trying to save model...")
    global agent
    global save
    agent.model.save(save)
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

action_space = DiscreteActions.create_actions()

save = input("Please enter name to save: ")

agent = GenModel(action_space=len(action_space), epsilon=0.9, gamma=0.99, epsilon_min=0.01, lr=0.0001, epsilon_decay=0.95, epochs=10, batch_size=1024)

env = gym.make("BipedalWalker-v3")

state = env.reset()
state = np.array(state)
# Explore 100% 100 games
print("Exploring 100 games")
for ii in range(100):
    print(str(round((ii+1)/100 * 100, 2)) + "% Done", end="\r", flush=True)
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action_index = random.randrange(len(action_space))
    action = np.array(action_space[action_index])
    next_state, reward, done, _ = env.step(action)
    next_state = np.array(next_state)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        agent.memory.append((state, action_index, reward, next_state))

        # Start new episode
        state = env.reset()
        state = np.array(state)
    else:
        # Add experience to memory
        agent.memory.append((state, action_index, reward, next_state))
        state = next_state

# memory = []
state = env.reset()
state = np.array(state)

# Train with Exploration decline to 1% 300 games
print("Exploring 1000 games with declaining epsilon")
for ep in range(1000):
    tot_reward = 0

    # states = []
    # actions = []
    # rewards = []

    done = False
    # Moves in game
    while not done:

        if agent.epsilon > np.random.rand():
            # Make a random action
            # action = env.action_space.sample()
            action_index = random.randrange(len(action_space))
            action = np.array(action_space[action_index])
        else:
            # Get action from Q-network
            Qs = agent.model(np.reshape(state, [1, agent.state_space]), training=False)[0]
            action_index = np.argmax(Qs)
            # print(action_index, len(action_space), Qs[action_index])
            action = action_space[action_index]

        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state)
        tot_reward += reward

        # agent.train_step(state, action, reward, 0.25)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            agent.memory.append((state, action_index, reward, next_state))

            print('Episode: {} of 1000'.format(ep),
                  'Total reward: {}'.format(tot_reward),
                  'Explore P: {:.4f}'.format(agent.epsilon))

            # states.append(state)
            # actions.append(action_index)
            # rewards.append(reward)

            state = env.reset()
            state = np.array(state)

            agent.train_on_minibatch(3)
            # memory.append((states, actions, rewards))
            
        else:
            # states.append(state)
            # actions.append(action_index)
            # rewards.append(reward)
            agent.memory.append((state, action_index, reward, next_state))
            state = next_state

    if (ep + 1) % 10 == 0:
        agent.update_epsilon()
        try:
            print("Saving model")
            agent.model.save(save)   
        except:
            print("Save failed")

env.close()

print("Saving model")
agent.model.save(save)



