import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def Q_model():
    with tf.variable_scope("qmodel"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss


env = gym.make('CartPole-v0')
q_mod = Q_model()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
t = 0
for _ in xrange(100):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
print t / 100

