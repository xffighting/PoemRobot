#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import json
import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    # shape = sentences.shape
    shape = np.shape(sentences)
    sentences = np.reshape(sentences, [-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, dictionary, batch_size, num_steps):
    
    # get the length of batches
    len_batch = len(vocabulary) // batch_size

    # get train data's index
    x_index = index_data(vocabulary, dictionary)

    # get label's index, fill the last one with total length
    y_index = index_data(vocabulary[1:], dictionary)
    y_index=np.append(y_index,len(dictionary))

    # prepare x y bathces
    x_batches = np.zeros([batch_size, len_batch], dtype=np.int32)
    y_batches = np.zeros([batch_size, len_batch], dtype=np.int32)

    for i in range(batch_size):
        x_batches[i] = x_index[len_batch*i : len_batch*(i+1)]
        y_batches[i] = y_index[len_batch*i : len_batch*(i+1)]

    # epoch size
    epoch_size = len_batch // num_steps

    for i in range(epoch_size):
        x = x_batches[:, num_steps*i : num_steps*(i+1)]
        y = y_batches[:, num_steps*i : num_steps*(i+1)]
        yield(x, y)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # 将字典保存到文件
    f=open('dictionary.json', 'w')
    f.write(json.dumps(dictionary))
    f.close()
    # 保存转置字典到文件
    f=open('reverse_dictionary.json', 'w')
    f.write(json.dumps(reversed_dictionary))
    f.close()
    return data, count, dictionary, reversed_dictionary
