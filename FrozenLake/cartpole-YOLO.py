import gym
import numpy as np
import tensorflow as tf
import random
import math

# initialize stuff
def Q_network():
  # neural network for Q(s,a)
  with tf.variable_scope("Q"):
    states = tf.placeholder(tf.float32, [None,4])
    W = tf.Variable(tf.zeros([4,2]))
    b = tf.Variable(tf.zeros([2]))
    Qout = tf.matmul(states, W) + b # returns 2 values; 1 for each action
    probabilities = tf.nn.softmax(Qout)
  
    # update the neural network based on new information
    newQ = tf.placeholder(tf.zeros([1,2])) # Q of each action
    loss = tf.reduce_sum(tf.square(newQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    updateModel = trainer.minimize(loss)
    
    return probabilities, states, Qout, newQ, updateModel
  
def run_episode(env, sess, network):
  # initialize any variables we need
  probs, states, Qout, newQ, updateModel = network
  observation = env.reset()
  totalreward = 0
  
  # run each episode
  for i in xrange(200):
    # observe the current state
    
    # converts input obs into a 2D array, to match the tensor format
    obs_vector = np.expand_dims(observation, axis=0)
    
    # pick an action based on the current policy
    calc_probs, current_Q = sess.run([probs, Qout],feed_dict = {states: obs})
    action = 0 if random.uniform(0,1) < calc_probs[0][0] else 1
    
    # take action, observe the next state and the reward we get
    obs, reward, done, info = env.step(action)
    totalreward += reward
    
    # Update Q(s,a)
    # First, find the max Q' value (find Q' for both actions, then take the max)
    next_Q = sess.run(Qout, feed_dict = {states:obs})
    maxQ = np.max(next_Q)
    
    # Then, update the current (state, action)'s Q value using the max Q' value
    new_Q = reward + 0.9*maxQ
#    target_Qs = np.expand_dims(current_Q, axis=0)
    target_Qs[action] = new_Q
    sess.run(updateModel, feed_dict = {newQ: target_Qs})
    
    # if we reach a terminal state, break and start a new episode
    if done:
      break



def main():
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  network = Q_network()
  
  print "network initialized"
  
  # run 2000 episodes
  for i in xrange(1):
    reward = run_episode(env, sess, network)
    print reward
    
    if reward == 200:
        print "reward 200 in " + str(i) + " iterations"
        break

    print "done"

main()

'''
  t = 0
  for _ in xrange(1):
    # reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
  print t / 100
'''