"""
Dueling Double DQN
Zhiang Chen, Jan 3 2018
MIT License
"""

import tensorflow as tf
import numpy as np

class Dueling_DDQN(object):
    def __init__(self,
                 n_action,
                 n_feature,
                 learning_rate,
                 batch_size,
                 gamma,
                 e_greedy
                 ):
        """
        1. get hyperparameters
        2. set placeholders
        3. build networks
        4. build an optimizer
        5. save the model
        6. initialize a session
        """
        # 1. get hyperparameters
        self.n_action = n_action
        self.n_feature = n_feature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.e_greedy = e_greedy

        # 2. set placeholders
        self.state = tf.placeholder(tf.float32, [self.batch_size, n_feature], name='state')
        self.state_ = tf.placeholder(tf.float32, [self.batch_size, n_feature], name='state_')
        self.G = tf.placeholder(tf.float32, [self.batch_size, 1], 'return')
        self.action = tf.placeholder(tf.int32, [self.batch_size, 1], 'action')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # 3. build networks
        self.beh_a_value, self.beh_s_value, self.beh_adv = self._build_network(input=self.state, scope='behavior')
        self.tar_a_value, self.tar_s_value, self.tar_adv = self._build_network(input=self.state_, scope='target')
        t_params = tf.get_collection('target')
        b_params = tf.get_collection('behavior')
        self.replace_target_op = [tf.assign(t, b) for t, b in zip(t_params, b_params)]

        # 4. build an optimizer
        self.loss, self.opt = self._build_optimizer()

        # 5. save the model
        self.saver = tf.train.Saver()

        # 6. initialize a session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("logs/", self.sess.graph)
        # tensorboard --logdir=./logs


    def _build_network(self, input, scope, trainable=True):
        hid_num1 = 200
        hid_num2 = 20
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(input, hid_num1, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, hid_num2, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            adv = tf.layers.dense(hidden2, self.n_action, activation=None, name='advantages', trainable=trainable,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden1_ = tf.layers.dense(input, hid_num1, activation=tf.nn.relu, name='fc1_', trainable=trainable,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            s_value = tf.layers.dense(hidden1_, 1, activation=None, name='state_value', trainable=trainable,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('action_value'):
                a_value = s_value + (adv - tf.reduce_mean(adv, axis=1, keep_dims=True))

            return a_value, s_value, adv

    def _build_optimizer(self):
        with tf.variable_scope('optimization'):
            with tf.variable_scope('loss'):
                batch_index = tf.range(self.batch_size, dtype=tf.int32)
                batch_index = tf.reshape(batch_index, [-1,1])
                indices = tf.concat([batch_index, self.action], axis=1)
                Q = tf.gather_nd(self.beh_a_value, indices)
                Q = tf.reshape(Q,[-1,1])
                loss = tf.reduce_mean(tf.squared_difference(self.G, Q))
            with tf.variable_scope('optimizer'):
                opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        return loss, opt

    def choose_beh_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.e_greedy:  # choosing action
            actions_value = self.sess.run(self.beh_a_value, feed_dict={self.state: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def choose_tar_action(self, state):
        state = state[np.newaxis, :]
        actions_value = self.sess.run(self.tar_a_value, feed_dict={self.state: state})
        return np.argmax(actions_value)

    def learn(self, state, action, G):
        state = state[np.newaxis, :]
        los, _ = self.sess.run([self.loss, self.opt], feed_dict={self.state: state, self.action: action, self.G: G})


    def save_model(self, loc='model'):
        save_path = self.saver.save(self.sess, "./" + loc + "/model.ckpt")
        print("Model saved in file: %s" % save_path)

    def restore_model(self, loc='model'):
        print("Restored model")
        self.saver.restore(self.sess, "./" + loc + "/model.ckpt")

    def replace_target(self):
        self.sess.run(self.replace_target_op)



if __name__ == '__main__':
    net = Dueling_DDQN(n_action=8, n_feature=6, learning_rate=0.001,
                       batch_size=64, gamma=0.9, e_greedy=0.9)
