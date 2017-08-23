#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
   A Deep CNN Network for QA of insurance fields
   Reference from this paper: "APPLYING DEEP LEARNING TO ANSWER SELECTION:A STUDY AND AN OPEN TASK"
   
"""

from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 Xuming Lin. All Rights Reserved"
__author__    = "Xuming Lin, Hai Liang Wang<hailiang.hl.wang@gmail.com>"
__date__      = "2017-08-21:23:30:05"


import tensorflow as tf
import numpy as np
from model import Model

# QA的CNN网络,自底向上为:
# word embedding
# tanh隐藏层
# convolution+tanh
# 1-max-pooling+tanh(Q和A分开)
# 计算cosine
class QACNN(Model):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, aplus_embed, aminus_embed = self.add_embeddings()
        # [batch_size, sequence_size, hidden_size, 1]
        self.h_q, self.h_ap, self.h_am = self.add_hl(q_embed, aplus_embed, aminus_embed)
        # [batch_size, total_channels]
        real_pool_q, real_pool_ap, real_pool_am = self.add_model(self.h_q, self.h_ap, self.h_am)
        # [batch_size, 1]
        self.q_ap_cosine, self.q_am_cosine = self.calc_cosine(real_pool_q, real_pool_ap, real_pool_am)
        # 损失和精确度
        self.total_loss, self.loss, self.accu = self.add_loss_op(self.q_ap_cosine, self.q_am_cosine)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    # 输入
    def add_placeholders(self):
        # 问题
        self.q = tf.placeholder(np.int32,
                shape=[None, self.config.sequence_length],
                name='Question')
        # 正向回答
        self.aplus = tf.placeholder(np.int32,
                shape=[None, self.config.sequence_length],
                name='PosAns')
        # 负向回答
        self.aminus = tf.placeholder(np.int32,
                shape=[None, self.config.sequence_length],
                name='NegAns')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # word embeddings
    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            embeddings = tf.get_variable('embeddings', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            aplus_embed = tf.nn.embedding_lookup(embeddings, self.aplus)
            aminus_embed = tf.nn.embedding_lookup(embeddings, self.aminus)
            return q_embed, aplus_embed, aminus_embed

    # Hidden Layer
    def add_hl(self, q_embed, aplus_embed, aminus_embed):
        with tf.variable_scope('HL'):
            W = tf.get_variable('weights', shape=[self.config.embedding_size, self.config.hidden_size], initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.config.hidden_size]))
            h_q = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(q_embed, [-1, self.config.embedding_size]), W)+b), [self.config.batch_size, self.config.sequence_length, -1])
            h_ap = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aplus_embed, [-1, self.config.embedding_size]), W)+b), [self.config.batch_size, self.config.sequence_length, -1])
            h_am = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aminus_embed, [-1, self.config.embedding_size]), W)+b), [self.config.batch_size, self.config.sequence_length, -1])
            tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(W))
            return h_q, h_ap, h_am

    # CNN层
    def add_model(self, h_q, h_ap, h_am):
        pool_q = list()
        pool_ap = list()
        pool_am = list()
        h_q = tf.reshape(h_q, [-1, self.config.sequence_length, self.config.hidden_size, 1])
        h_ap = tf.reshape(h_ap, [-1, self.config.sequence_length, self.config.hidden_size, 1])
        h_am = tf.reshape(h_am, [-1, self.config.sequence_length, self.config.hidden_size, 1])
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # filter的W和b
                conv1_W = tf.get_variable('W', shape=[filter_size, self.config.hidden_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv1_b = tf.get_variable('conv_b', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # pooling层的bias,Q和A分开
                pool_qb = tf.get_variable('pool_qb', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                pool_ab = tf.get_variable('pool_ab', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # 卷积
                out_q = tf.nn.relu((tf.nn.conv2d(h_q, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                # 池化
                out_q = tf.nn.max_pool(out_q, [1,self.config.sequence_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_q = tf.nn.tanh(out_q+pool_qb)
                pool_q.append(out_q)

                out_ap = tf.nn.relu((tf.nn.conv2d(h_ap, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                out_ap = tf.nn.max_pool(out_ap, [1,self.config.sequence_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_ap = tf.nn.tanh(out_ap+pool_ab)
                pool_ap.append(out_ap)

                out_am = tf.nn.relu((tf.nn.conv2d(h_am, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                out_am = tf.nn.max_pool(out_am, [1,self.config.sequence_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_am = tf.nn.tanh(out_am+pool_ab)
                pool_am.append(out_am)

                # 加入正则项
                tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(conv1_W))

        total_channels = len(self.config.filter_sizes)*self.config.num_filters

        real_pool_q = tf.reshape(tf.concat(pool_q, 3), [-1, total_channels])
        real_pool_ap = tf.reshape(tf.concat(pool_ap, 3), [-1, total_channels])
        real_pool_am = tf.reshape(tf.concat(pool_am, 3), [-1, total_channels])

        return real_pool_q, real_pool_ap, real_pool_am

    # 计算cosine
    def calc_cosine(self, real_pool_q, real_pool_ap, real_pool_am):
        len_pool_q = tf.sqrt(tf.reduce_sum(tf.pow(real_pool_q, 2), [1]))
        len_pool_ap = tf.sqrt(tf.reduce_sum(tf.pow(real_pool_ap, 2), [1]))
        len_pool_am = tf.sqrt(tf.reduce_sum(tf.pow(real_pool_am, 2), [1]))

        q_ap_cosine = tf.div(tf.reduce_sum(tf.multiply(real_pool_q, real_pool_ap), [1]), tf.multiply(len_pool_q, len_pool_ap))
        q_am_cosine = tf.div(tf.reduce_sum(tf.multiply(real_pool_q, real_pool_am), [1]), tf.multiply(len_pool_q, len_pool_am))

        return q_ap_cosine, q_am_cosine

    # 损失节点
    def add_loss_op(self, q_ap_cosine, q_am_cosine):
        # margin值,论文用的0.009
        margin = tf.constant(self.config.m, shape=[self.config.batch_size], dtype=tf.float32)
        # 0常量
        zero = tf.constant(0., shape=[self.config.batch_size], dtype=tf.float32)
        l = tf.maximum(zero, tf.add(tf.subtract(margin, q_ap_cosine), q_am_cosine))
        loss = tf.reduce_sum(l)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        accu = tf.reduce_mean(tf.cast(tf.equal(zero, l), tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accu)
        return total_loss, loss, accu

    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op