#!/usr/bin/python
"""
Implement Deep Q Network learning algorithm By Using Tensorflow
Q-Network => DQN as following improvements
1. going from single-layer network to a multi-layer convolutional network.
2. implementing experience replay, train network using stored memories from it's experience.
3. utilizing a second "target" network to compute target Q-values during updates.
Create By coder-james
"""
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
from gridworld import gameEnv

env = gameEnv(partial=False, size=5)

class Qnetwork():
    def __init__(self, h_size):
        #recieve a frame from the game, flattened into an array. Then resize it and process it through four convolutinal layers
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32) # [84 x 84 x 3]
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        #conv1 [-1, 20, 20, 32]
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        #conv2 [-1, 9, 9, 64]
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        #conv3 [-1, 7, 7, 64]
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)
        #conv4 [-1, 1, 1, 512]

        self.streamAC, self.streamVC = tf.split(3, 2, self.conv4) #split into 2 parts by dimension 3
        #streamAC/streamVC [-1, 1, 1, 256]
        self.streamA = tf.contrib.layers.flatten(self.streamAC) #[-1, 256]
        self.streamC = tf.contrib.layers.flatten(self.streamVC) #[-1, 256]
        self.AW = tf.Variable(tf.random_normal([h_size/2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size/2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW) #[-1, 4]
        self.Value = tf.matmul(self.streamV, self.VW) #[-1, 1]
        #combine together to get final Q-values
        self.Qout = self.Value + tf.sub(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices = 1, keep_dims=True)) # value - (q - E[q])
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer_size)), [size, 5])

def processState(state):
    return np.reshape(states, [21168])
#update the parameters of target network with those of the primary network
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0: total_vars / 2]):
        op_holder.append(tfVars[idx + total_vars / 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars / 2].value())))
    return op_holder
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

batch_size = 32 # how many experiences to use for each training step.
update_freq = 4 # how often to perform a training step.
y = .99 # discount factor on the target Q-values
startE = 1 # starting chance of random action
endE = .1 # final chance of random action
anneling_steps = 10000. #steps of training to reduce startE to endE
num_episodes = 10000
pre_train_steps = 10000 #steps of random actions before training begins
max_epLength = 50 #max allowed length of episode
load_model = False
path = "./dqn" #save model
h_size = 512
tau = 0.001 # rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)
init = tf.initialize_all_variables()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

e = startE
stepDrop = (startE - endE) / anneling_steps
jList = [] #steps per episode
rList = [] #total rewards
total_steps = 0
# if not os.path.exists(path):
    # os.makedirs(path)
with tf.Session as sess:
    if load_model:
        print "loading model..."
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps, sess) #set target network to be equal to the primary netowrk
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        observation = env.reset()
        state = processState(observation)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:
            j += 1
            #choose an action by greedily(with e chance of random action)
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                action = np.random.randint(0, 4)
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
            state1, reward, done = env.step(action)
            state1 = processState(state1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([state, action, reward, state1, done]),[1,5]))
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)
            rAll += reward
            state = state1
            if done:
                break
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # if i % 100 == 0:
            # saver.save(sess, path + "/model-" + str(i) + ".cptkd")
            # print "Saved Model"
        if len(rList) % 10 = 0:
            print total_steps, np.mean(rList[-10:]), e
    # saver.save(sess, path + '/model-' + str(i) + ".cptk")
print "percent of successful episodes: " + str(sum(rList)/num_episodes) + "%"

rMat = np.resize(np.array(rList), [len(rList) / 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)
# plt.show()