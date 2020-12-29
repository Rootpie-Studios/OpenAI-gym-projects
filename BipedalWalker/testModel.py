import gym
import numpy as np
import keras
import DiscreteActions

env = gym.make("BipedalWalker-v3")

state = env.reset()

name = input("Please state model to load: ")

model = keras.models.load_model(name)
action_space = DiscreteActions.create_actions()

for ep in range(10):
    done = False

    while not done:
        env.render()
        Qs = model(np.reshape(state, [1, 24]), training=False)[0]
        action_index = np.argmax(Qs)
        action = action_space[action_index]

        next_state, reward, done, info = env.step(action)


        if done:
            state = env.reset()
        else:
            state = next_state
            state = np.array(state)
            

env.close()