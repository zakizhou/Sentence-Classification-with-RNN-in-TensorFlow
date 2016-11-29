from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf
import numpy as np
import re
import codecs
from nltk.tokenize import word_tokenize
import collections
from sklearn.model_selection import StratifiedKFold


def clean_sentence(sentence):
    """
    Tokenization/sentence cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()


def build_corpus():
    positive_sentences = codecs.open("rt-polaritydata/rt-polarity.pos").readlines()
    negative_sentences = codecs.open("rt-polaritydata/rt-polarity.neg").readlines()
    num_positive = len(positive_sentences)
    num_negative = len(negative_sentences)
    labels = [1] * num_positive + [0] * num_negative
    sentences = positive_sentences + negative_sentences
    clean = [word_tokenize(clean_sentence(sentence)) for sentence in sentences]
    total = reduce(lambda sent1, sent2: sent1 + sent2, clean)
    counter = collections.Counter(total)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word2id = dict(zip(words, range(3, len(words)+3)))
    word2id["<pad>"] = 0
    word2id["<sos>"] = 1
    word2id["<eos>"] = 2
    inputs = []
    for sent in clean:
        stantard_sent = [1] + [word2id[word] for word in sent] + [2]
        inputs.append(stantard_sent)
    skf = StratifiedKFold(n_splits=5)
    inputs_array = np.array(inputs)
    labels_array = np.array(labels)
    train_index, validation_index = skf.split(inputs_array, labels_array).next()
    np.random.shuffle(train_index)
    np.random.shuffle(validation_index)
    train_X, train_y = inputs_array[train_index], labels_array[train_index]
    valid_X, valid_y = inputs_array[validation_index], labels_array[validation_index]
    return word2id, train_X, train_y, valid_X, valid_y


def arr2str(arr):
    return np.array(arr).tostring()


def convert_to_records(sequences, labels, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for sequence, label in zip(sequences, labels):
        example = tf.train.Example(features=tf.train.Features(feature={
            "sequence": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(sequence)])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sequence)]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    word2id, train_sequences, train_labels, valid_sequences, valid_labels = build_corpus()
    convert_to_records(train_sequences, train_labels, "tfrecords/train.tfrecords")
    convert_to_records(valid_sequences, valid_labels, "tfrecords/valid.tfrecords")


if __name__ == "__main__":
    main()
