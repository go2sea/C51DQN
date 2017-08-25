# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
from collections import deque
from Config import C51DQNConfig
from myUtils import lazy_property, conv, dense
import math


class C51DQN:
    def __init__(self, env, config):
        self.sess = tf.InteractiveSession()
        self.config = config
        self.v_max = self.config.v_max
        self.v_min = self.config.v_min
        self.atoms = self.config.atoms
        self.time_step = 0
        self.atoms = self.config.atoms
        self.epsilon = self.config.INITIAL_EPSILON
        self.state_shape = env.observation_space.shape
        self.action_dim = env.action_space.n

        target_state_shape = [1]
        target_state_shape.extend(self.state_shape)
        self.state_input = tf.placeholder(tf.float32, target_state_shape)
        self.action_input = tf.placeholder(tf.int32, [1, 1])

        self.m_input = tf.placeholder(tf.float32, [self.atoms])

        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]

        self.p

        self.cross_entropy_loss
        self.optimize

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.save_model()
        self.restore_model()

    @lazy_property
    def p(self):
        c_names = [tf.GraphKeys.GLOBAL_VARIABLES]
        w_i = tf.random_uniform_initializer(-0.1, 0.1)
        b_i = tf.constant_initializer(0.1)
        return self.build_net(self.state_input, self.action_input, c_names, 24, 24, w_i, b_i)

    def build_net(self, state, action, c_names, units_1, units_2, w_i, b_i, reg=None):
        with tf.variable_scope('conv1'):
            conv1 = conv(state, [5, 5, 3, 6], [6], [1, 2, 2, 1], w_i, b_i)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, [3, 3, 6, 12], [12], [1, 2, 2, 1], w_i, b_i)
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(conv2)
            # 两种reshape写法
            # flatten = tf.reshape(relu5, [-1, np.prod(relu5.get_shape().as_list()[1:])])
            # flatten = tf.reshape(relu5, [-1, np.prod(relu5.shape.as_list()[1:])])
            # print flatten.get_shape()
        with tf.variable_scope('dense1'):
            dense1 = dense(flatten, units_1, [units_1], w_i, b_i)
        with tf.variable_scope('dense2'):
            dense2 = dense(dense1, units_2, [units_2], w_i, b_i)
        with tf.variable_scope('concat'):
            concatenated = tf.concat([dense2, tf.cast(action, tf.float32)], 1)
        with tf.variable_scope('dense3'):
            dense3 = dense(concatenated, self.atoms, [self.atoms], w_i, b_i)
        return dense3

    @lazy_property
    def Q(self):
        return tf.reduce_sum(self.z * self.p)

    @lazy_property
    def cross_entropy_loss(self):
        return -tf.reduce_sum(self.m_input * tf.log(self.p))

    def train(self, s, r, action, s_, gamma):
        list_Q_ = [self.Q.eval(feed_dict={self.state_input: [s_], self.action_input: [[a]]}) for a in range(self.action_dim)]
        a_ = tf.argmax(list_Q_).eval()
        m = np.zeros(self.atoms)
        p = self.p.eval(feed_dict={self.state_input: [s_], self.action_input: [[a_]]})[0]
        for j in range(self.atoms):
            Tz = min(self.v_max, max(self.v_min, r + gamma * self.z[j]))
            bj = (Tz - self.v_min) / self.delta_z
            l, u = math.floor(bj), math.ceil(bj)
#            pj = self.p.eval(feed_dict={self.state_input: [s_], self.action_input: [[a_]]})[0][j]
            pj = p[j]
            m[int(l)] += pj * (u - bj)
            m[int(u)] += pj * (bj - l)
        self.sess.run(self.optimize, feed_dict={self.state_input: [s], self.action_input: [action], self.m_input: m})

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        return optimizer.minimize(self.cross_entropy_loss)  # optimizer只更新selese_network中的参数

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.config.MODEL_PATH))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.MODEL_PATH)
        print("Model restored.")

    def greedy_action(self, s):
        self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)
        print 'self.epsilon:', self.epsilon
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        print 'self.action_dim:', self.action_dim
        return np.argmax([self.Q.eval(feed_dict={self.state_input: [s], self.action_input: [[a]]}) for a in range(self.action_dim)])
