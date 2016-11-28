#!/usr/bin/python
import gym
import numpy as np

"""
Implement Q-Table learning algorithm
Create By coder-james
Star from Github Gist
"""

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = .85 #learning rate
y = .99 #discounted coefficient
num_episodes = 1000
rList = [] #total rewards
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    j = 0
    while j < 100:
        j += 1
        #choose an action by greedily(with noise)
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        state1, reward, done, _ = env.step(action) #'_' means 'info'
        Q[state, action] = Q[state, action] + lr * (reward + y * np.max(Q[state1, :]) - Q[state, action])
        rAll += reward
        state = state1
        if done:
            break
    rList.append(rAll)
print "Score over times: %s" %(sum(rList) / num_episodes)
print "Final Q-Table Values" 
print Q