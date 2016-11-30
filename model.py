from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class Config(object):
    def __init__(self):
        self.num_units = 96
        self.learning_rate = 5e-4
        self.embedding_size = 128
        self.hidden_size = 256
        self.num_sampled = 10


class TextModel(object):
    def __init__(self, config, inputs):
        num_units = config.num_units
        lstm = rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        vocab_size = inputs.vocab_size
        embedding_size = config.embedding_size
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        shape=[vocab_size, embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            embed = tf.nn.embedding_lookup(embedding, inputs.inputs)
        with tf.variable_scope("rnn"):
            _, states = rnn.dynamic_rnn(cell=lstm,
                                        inputs=embed,
                                        sequence_length=inputs.sequence_length,
                                        dtype=tf.float32)
        with tf.variable_scope("hidden"):
            weights = tf.get_variable(name="weights",
                                      shape=[num_units, config.hidden_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.05),
                                      dtype=tf.float32)
            bias = tf.get_variable(name="bias",
                                   shape=[config.hidden_size],
                                   initializer=tf.constant_initializer(value=0.),
                                   dtype=tf.float32)
            xw_plus_b = tf.nn.xw_plus_b(states[1], weights, bias)
            output = tf.nn.relu(xw_plus_b)
        with tf.variable_scope("nce_loss"):
            output_embedding = tf.get_variable("output_embedding",
                                               shape=[inputs.num_classes, config.hidden_size],
                                               initializer=tf.truncated_normal_initializer(stddev=0.05),
                                               dtype=tf.float32)
            output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                    shape=[inputs.num_classes],
                                                    initializer=tf.constant_initializer(value=0.),
                                                    dtype=tf.float32)
            loss_per_example = tf.nn.nce_loss(weights=output_embedding,
                                              biases=output_embedding_bias,
                                              inputs=output,
                                              num_classes=inputs.num_classes,
                                              num_true=1,
                                              num_sampled=config.num_sampled)
            self.__loss = tf.reduce_mean(loss_per_example)
        with tf.variable_scope("output"):
            softmax_w = tf.get_variable(name="softmax_w",
                                        shape=[config.hidden_size, inputs.num_classes],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(name="softmax_b",
                                        shape=[inputs.num_classes],
                                        initializer=tf.constant_initializer(value=0.05),
                                        dtype=tf.float32)
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        with tf.name_scope("train"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inputs.labels)
            self.__loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)
        with tf.name_scope("validation"):
            predict = tf.argmax(logits, 1)
            equal = tf.equal(predict, inputs.labels)
            self.__accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op

    @property
    def validation(self):
        return self.__accuracy
