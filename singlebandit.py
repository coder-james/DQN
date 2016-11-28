#!/usr/bin/python
import tensorflow as tf
import numpy as np
"""
Implement Single bandit learning algorithm By Using Tensorflow
Create By coder-james
"""

#List out bandits.
bandits = [.2, 0, -.2, -5]
num_bandits = len(bandits)

def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1 #positive reward
    else:
        return -1

tf.reset_default_graph()

W = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(W, 0)

reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(W, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.1

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])
        _, resp, w = sess.run([update, responsible_weight, W], feed_dict={reward_holder: [reward], action_holder: [action]})
        total_reward[action] += reward
        if i % 50 == 0:
            print "Running reward for the %s bandits: %s" %(num_bandits, total_reward)
        i += 1
print "The agent thinks bandit %s is the most promising.." % (np.argmax(w) + 1)
print w
if np.argmax(w) == np.argmax(-np.array(bandits)):
    print "right"
else:
    print "false"