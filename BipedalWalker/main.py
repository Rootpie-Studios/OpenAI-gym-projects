import gym
import numpy as np 

env = gym.make("BipedalWalker-v3")

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, _ = env.step(action)

print(env.action_space)
print(env.observation_space)
print("Action: ", action)
print("Reward: ", reward)

for ii in range(1):
    done = False

    while not done:
        # env.render()

        action = env.action_space.sample()
        print(action)
        next_state, reward, done, _ = env.step(action)

        if done:
            # Start new episode
             state = env.reset()

env.close()