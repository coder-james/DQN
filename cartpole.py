#!/usr/bin/python
"""
Implement Cart-Pole learning algorithm By Using Tensorflow
Policy-Based Algorithm
Cart-Pole Task:
    Have an agent learn how to balance a pole for as long as possible without it falling.
Unlike the two-armed bandit, this task requires: Observations and Delayed reward
Create By coder-james
"""
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

env = gym.make('CartPole-v0')
env.reset()
"""
randomly choosing actions can only get a couple dozen rewards
"""
# random_episodes = 0
# reward_sum = 0
# while random_episodes < 10:
#     env.render()
#     observation, reward, done, _ = env.step(np.random.randint(env.action_space.n))
#     reward_sum += reward
#     if done:
#         random_episodes += 1
#         print "Reward for this episode was: %s" % reward_sum
#         reward_sum = 0
#         env.reset()

H = 10 # number of hidden layer neurons
batch_size = 50 # every how many episodes to do a param update
learning_rate = 1e-2 #play with this to train faster or more stably
gamma = 0.99 # discount factor for reward
D = env.observation_space.shape[0] #input dimensionality (4 for cartpole)

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score) #giving a probablity of chosing to the action of moving left or right.

tvars = tf.trainable_variables() #get W1 and W2 variable
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

loglike = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglike * advantages)
newGrads = tf.gradients(loss, tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))  #apply gradient descents update to (W1Grad, W1),(W2Grad, W2)

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs, ys, drs = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    while episode_number <= total_episodes:
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation, [1, D])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [],[],[]
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            if episode_number % batch_size == 0: #batch update Weights values using All Gradients of W1 and W2
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print "Average reward for episode %f. Total average reward %f." %(reward_sum / batch_size, running_reward / batch_size)
                if reward_sum / batch_size > 200:
                    print "Task solved in %s episodes!" % episode_number
                    break
                reward_sum = 0
            observation = env.reset()
print episode_number, "Episodes completed"