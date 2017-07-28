# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import C51DQNConfig
from C51 import C51DQN
import random

def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def BreakOut_C51DQN(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = C51DQN(env, C51DQNConfig())
    scores = []
    for e in range(C51DQNConfig.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        # while done is False:
        last_lives = 5
        throw = True
        while not done:
            env.render()
            action = 1 if throw else agent.greedy_action(state)
            print 'action:', action
            next_state, real_reward, done, info = env.step(action)
            lives = info['ale.lives']
            train_reward = 1 if throw else -1 if lives < last_lives else real_reward
            score += real_reward
            throw = lives < last_lives
            last_lives = lives
            agent.train(state, train_reward, [action], next_state, 0.1)
            state = next_state
        scores.append(score)
        print "episode:", e, "  score:", score
        # if np.mean(scores[-min(10, len(scores)):]) > 495:
        #     break
    return scores


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    # env = gym.make(NoisyNetDQNConfig.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    NoisyNetDQN_sum_scores = np.zeros(C51DQNConfig.episode)
    for i in range(C51DQNConfig.iteration):
        scores = BreakOut_C51DQN(i, env)
        c51_sum_scores = [a + b for a, b in zip(scores, NoisyNetDQN_sum_scores)]
    NoisyNetDQN_mean_scores = NoisyNetDQN_sum_scores / C51DQNConfig.iteration
    with open('/Users/mahailong/C51DQN/NoisyNetDQN_mean_scores.p', 'wb') as f:
        pickle.dump(NoisyNetDQN_mean_scores, f, protocol=2)

    # map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
        # xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    # env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


