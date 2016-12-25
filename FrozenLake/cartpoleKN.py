import gym
import numpy as np
import tensorflow
env = gym.make('CartPole-v0')


def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:       
		    return totalreward

for i_episode in range(20):
    parameters = np.random.rand(4) * 2 - 1  
    reward = run_episode(env, parameters)
    print(parameters)
    print(reward)

