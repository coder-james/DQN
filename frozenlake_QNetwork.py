#!/usr/bin/python
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Implement QNetwork learning algorithm By Using Tensorflow
Create By coder-james
Star from Github Gist
"""

env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

input = tf.placeholder(shape=[1, env.observation_space.n], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([env.observation_space.n, env.action_space.n], 0, 0.01))
Qout = tf.matmul(input, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, env.action_space.n], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

y = .99 #discounted coefficient
e = 0.1
num_episodes = 2000
jList = [] #steps per episode
rList = [] #total rewards
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        state = env.reset()
        rAll = 0
        done = False
        j = 0
        while j < 100:
            j += 1
            #choose an action by greedily(with noise)
            action, allQ = sess.run([predict, Qout], feed_dict={input: np.identity(env.observation_space.n)[state: state + 1]}) #identity matrix fetch one row
            if np.random.rand(1) < 0.1:
                action[0] = env.action_space.sample()
            state1, reward, done, _ = env.step(action[0]) #'_' means 'info'
            Q1 = sess.run(Qout, feed_dict={input: np.identity(env.observation_space.n)[state1: state1 + 1]})
            targetQ = allQ
            targetQ[0, action[0]] = reward + y * np.max(Q1)
            _, W1 = sess.run([updateModel, W], feed_dict={input: np.identity(env.observation_space.n)[state: state + 1], nextQ: targetQ})
            rAll += reward
            state = state1
            if done:
                e = 1./((i / 50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Score over times: %s" %(sum(rList) / num_episodes)
plt.plot(rList)
plt.show()