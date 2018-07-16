# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import logging 

class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float 0.001
            学习率.
        batch_size : int 16 
            batch_size.
        num_steps : int 32
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int 5000
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int 128
            embding中，编码后的字向量的维度
        rnn_layers : int 3
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
                self.lstm_inputs = tf.nn.embedding_lookup(embed, self.X)
                print("lstm_inputs:",self.lstm_inputs.shape)
            else:
                # if not, initialize an embedding and train it.
                with tf.device("/cpu:0"):
                    embed = tf.get_variable('embedding', [self.num_words, self.dim_embedding])
                    # tf.summary.histogram('embed', embed)
                    print("embed:",embed.shape)
                    self.lstm_inputs = tf.nn.embedding_lookup(embed, self.X)
                    print("lstm_inputs:",self.lstm_inputs.shape)

        with tf.variable_scope('rnn'):

            # basic cell
            def lstm_cell():
                lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_embedding,forget_bias=0, state_is_tuple=True)
                drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
                return drop

            # multi sell
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.rnn_layers)] ,state_is_tuple=True)
         
        
            # init state
            self.init_state = cell.zero_state(self.batch_size, tf.float32)

          
            lstm_outputs_tensor, state = tf.nn.dynamic_rnn(cell, inputs=self.lstm_inputs, initial_state=self.init_state, time_major=False)
           
            final_outputs_tensor = lstm_outputs_tensor[:, -1, :]
            self.outputs_state_tensor = final_outputs_tensor
            self.final_state = state

        # print("lstm_outputs_tensor:", lstm_outputs_tensor.shape)
        # 沿着batch_size展开
        seq_output = tf.concat(lstm_outputs_tensor, 1)
        # print("seq_output:",seq_output.shape)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])
        # print("seq_output_final:",seq_output_final.shape)

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", 
                                        [self.dim_embedding, self.num_words],
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            softmax_b = tf.get_variable("softmax_b", 
                                        [self.num_words],
                                        initializer=tf.constant_initializer(0.0))

            

            # print("softmax_w:",softmax_w.shape)
            # print("softmax_b:",softmax_b.shape)
            logits = tf.matmul(seq_output_final, softmax_w) + softmax_b
            # print("logits:",logits.shape)

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')
        print("logits:",logits.shape)
        # label one hot encoder
        y_one_hot = tf.one_hot(self.Y, self.num_words)
        print("y_one_hot:",y_one_hot.shape)
        labels = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)
        _,variance = tf.nn.moments(logits, -1)
        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(variance))
        tf.summary.scalar('var_loss', var_loss)
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)
        tf.summary.scalar('loss', self.loss)
        self.merged_summary_op = tf.summary.merge_all()