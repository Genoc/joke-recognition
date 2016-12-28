import gym
import numpy as np
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt

# hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.5

EPSILON = 0.1 # used in epsilon-greedy exploration policy
USE_GREEDY_EXPLORATION=False

MINIBATCH_SIZE = 10 # for experience replay

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
    newQ = tf.placeholder(tf.float32, [1,2]) # Q of each action
    loss = tf.reduce_sum(tf.square(newQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    updateModel = trainer.minimize(loss)
    
    return probabilities, states, Qout, newQ, updateModel
  
def run_episode(env, sess, network, history):
  # initialize any variables we need
  probs, states, Qout, newQ, updateModel = network
  observation = env.reset()
  totalreward = 0
  
  # run each episode
  for i in xrange(200):
    # converts the current observation into a 2D array, to match the tensor format
    obs_vector = np.expand_dims(observation, axis=0)
    
    # pick an action based on the current policy
    calc_probs = sess.run(probs, feed_dict = {states: obs_vector})
    
    if USE_GREEDY_EXPLORATION:
      if calc_probs[0][0] > calc_probs[0][1]:
        action = 0
      else:
        action = 1
        
      if random.uniform(0,1) < EPSILON:
        action = env.action_space.sample()
    else:
      action = 0 if random.uniform(0,1) < calc_probs[0][0] else 1
    
    # take action, observe the next state and the reward we get
    obs, reward, done, info = env.step(action)
    totalreward += reward

    # store the transition in our history
    # {old state, action, reward, new state}
    # We only store the most recent 500 observations (to keep the memory recent)
    if (len(history) >= 500):
        del history[0]
    history.append([observation, action, reward, obs])
    
    # update the current state to be the new one
    observation = obs
    
    # Experience replay: Update Q(s,a) for a sample of the history
    sample_states = random.sample(history, min(len(history), MINIBATCH_SIZE))
    # sample_states = [history[-1]]
    for iter in enumerate(sample_states):
      # get the sample state's info
      index, sample_state = iter
      old_state, action, reward, new_state = sample_state
      oldstate_vector = np.expand_dims(old_state, axis=0)
      newstate_vector = np.expand_dims(new_state, axis=0)
      
      # Obtain the Q values for the old state
      current_Q = sess.run(Qout, feed_dict = {states:oldstate_vector})
      
      # obtain the Q values for the new state
      next_Q = sess.run(Qout, feed_dict = {states:newstate_vector})
      maxQ = np.max(next_Q)
      
      # Then, update the current (state, action)'s Q value using the max Q' value
      new_Q = reward + GAMMA*maxQ
      target_Qs = current_Q
      target_Qs[0, action] = new_Q
      sess.run(updateModel, feed_dict = {newQ: target_Qs, states: oldstate_vector})
    
    # if we reach a terminal state, break and start a new episode
    if done:
      break

  return totalreward

def main():
  network = Q_network()
  env = gym.make('CartPole-v0')
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  history = [] 
  rew = []
    
  # run episodes
  for i in xrange(500):
    reward = run_episode(env, sess, network, history)
    print 'reward: ' + str(reward)
    rew.append(reward)
        
    if reward == 200:
        print "reward 200 in " + str(i) + " iterations"
        break

  plt.hist(rew, bins='auto')
  plt.show()

main()

'''
  t = 0
  for _ in xrange(1):
    # reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
  print t / 100
'''