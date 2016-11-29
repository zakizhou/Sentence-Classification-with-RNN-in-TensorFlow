from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized,
                                       features={
                                           "length": tf.FixedLenFeature([], tf.int64),
                                           "sequence": tf.FixedLenFeature([], tf.string),
                                           "label": tf.FixedLenFeature([], tf.int64)
                                       })
    length = features["length"]
    sequence = tf.decode_raw(features['sequence'], tf.int64)
    label = features['label']
    return sequence, label, length


def input_producer(batch_size, capacity, train=True):
    if train is True:
        filename_queue = tf.train.string_input_producer(["tfrecords/train.tfrecords"], num_epochs=20)
    else:
        filename_queue = tf.train.string_input_producer(["tfrecords/valid.tfrecords"], num_epochs=None)
    sequence, label, length = read_and_decode(filename_queue)
    sequences, labels, lengths = tf.train.batch([sequence, label, length],
                                                batch_size=batch_size,
                                                dynamic_pad=True,
                                                capacity=capacity)
    return sequences, labels, lengths


class Inputs(object):
    def __init__(self,
                 batch_size,
                 capacity,
                 train=True):
        self.batch_size = batch_size
        self.vocab_size = 18595
        self.num_classes = 2
        self.capacity = capacity
        self.inputs, self.labels, self.sequence_length = input_producer(batch_size=self.batch_size,
                                                                        capacity=self.capacity,
                                                                        train=train)
