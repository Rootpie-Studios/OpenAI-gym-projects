import gym
import random

env = gym.make('CartPole-v0')

policy = {}
avg1000 = 0
epv = 0.1

for episode in range(100000):
    observation = env.reset()

    actions = []

    for t in range(1000):
        # env.render()

        n0 = round(observation[0], 1)
        n1 = round(observation[1], 1)
        n2 = round(observation[2], 1)
        n3 = round(observation[3], 1)
        state = str(n0) + str(n1) + str(n2) + str(n3)

        if state in policy:
            action = policy[state].index(max(policy[state]))
            epsilon = random.uniform(0, 1)

            if epsilon > epv:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()


        observation, reward, done, info = env.step(action)

        actions.append([state, action])

        if done:

            for action in actions:
                if action[0] in policy:
                    policy[action[0]][action[1]] = 10 * (t + 1)
                else:
                    policy.update({action[0] : [0, 0]})
                    policy[action[0]][action[1]] = 10 * (t + 1)

            avg1000 += t
            if (episode + 1) % 1000 == 0:
                print("avg score 1000 episodes {}".format(avg1000/1000))
                epv = epv * 1.03
                avg1000 = 0

            break

observation = env.reset()
iterations = 0
found = 0
while True:
    env.render()

    n0 = round(observation[0], 1)
    n1 = round(observation[1], 1)
    n2 = round(observation[2], 1)
    n3 = round(observation[3], 1)
    state = str(n0) + str(n1) + str(n2) + str(n3)

    if state in policy:
        found += 1
        action = policy[state].index(max(policy[state]))
    else:
        action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    if done:
        print("We survied {} actions".format(iterations + 1))
        print("Found {} actions".format(found))
        break

    iterations += 1


env.close()