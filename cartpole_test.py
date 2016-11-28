#!/usr/bin/python
import gym
from gym import spaces

Episode = 1
Iteration=100

def checkspace():
    space = spaces.Discrete(8)
    x = space.sample()
    assert space.contains(x)
    assert space.n == 8

def checkCartPole():
    env = gym.make('CartPole-v0')
    print(env.action_space.n)
    print(env.observation_space.shape[0])

def process():
    env = gym.make('CartPole-v0')
    for i_episode in range(Episode):
        observation = env.reset()
        for t in range(Iteration):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

if __name__ == "__main__":
    # process()
    checkCartPole()
