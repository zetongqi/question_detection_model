import os
import sys
import tensorflow as tf
from dataset.process_raw import PROCESSED_TRAIN, PROCESSED_VAL
import csv
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
from mag_model.mag_model import MAG_FILE
from pymagnitude import *
import numpy as np


# deprecated
class DataWrapper:
    def __init__(self, datafile_path, mag_file_path, batch_size=32, max_seq_len=100):
        self.FILE_PATH = datafile_path
        self.MAG_FILE = mag_file_path
        self.BATCH_SIZE = batch_size
        self.MAX_SEQ_LEN = max_seq_len
        self.vectors = Magnitude(self.MAG_FILE)
        print("this version is deprecated, please use DataWrapperV2")

    # reads the processed data and return data matrices
    def read_data(self):
        X = []
        y = []
        tsvFile = csv.reader(open(self.FILE_PATH), delimiter="\t")
        for row in tsvFile:
            X.append(row[0])
            y.append(int(row[1] == "1"))
        self.X = X
        self.y = y

    # tokenization method
    def tokenize(self, sequence):
        return word_tokenize(sequence)

    # generator for tf.data.Dataset object
    def data_generator(self):
        def pad_zeros(x, max_seq_len):
            if x.shape[0] >= max_seq_len:
                return x[0:max_seq_len, :]
            else:
                return np.concatenate(
                    (x, np.zeros((max_seq_len - x.shape[0], x.shape[1]))), axis=0
                )

        for x, y in zip(self.X, self.y):
            yield pad_zeros(self.vectors.query(self.tokenize(x)), self.MAX_SEQ_LEN), y

    # returns tf.data.Dataset object for model.fit
    def get_dataset(self):
        dataset = (
            tf.data.Dataset.from_generator(
                self.data_generator,
                output_types=(tf.float32, tf.uint8),
                output_shapes=(
                    tf.TensorShape((self.MAX_SEQ_LEN, self.vectors.dim)),
                    tf.TensorShape([]),
                ),
            )
            .shuffle(buffer_size=2000)
            .batch(self.BATCH_SIZE)
        )
        return dataset


class DataWrapperV2:
    def __init__(self, datafile_path, mag_file_path, batch_size=32, max_seq_len=100):
        self.FILE_PATH = datafile_path
        self.MAG_FILE = mag_file_path
        self.BATCH_SIZE = batch_size
        self.MAX_SEQ_LEN = max_seq_len
        self.vectors = Magnitude(self.MAG_FILE)

    # reads the processed data and return data matrices
    def read_data(self):
        X = []
        y = []
        tsvFile = csv.reader(open(self.FILE_PATH), delimiter="\t")
        for row in tsvFile:
            X.append(row[0])
            y.append(int(row[1] == "1"))
        self.X = X
        self.y = y

    # tokenization method
    def tokenize(self, sequence):
        return word_tokenize(sequence)

    def prepare_dataset(self, training=True):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        if training:
            dataset.repeat()

        def _process_string(x):

            # x is numpy array
            def _pad_zeros(x, max_seq_len):
                if x.shape[0] >= max_seq_len:
                    return x[0:max_seq_len, :]
                else:
                    return np.concatenate(
                        (x, np.zeros((max_seq_len - x.shape[0], x.shape[1]))), axis=0
                    )

            x = x.numpy().decode()
            x = self.tokenize(x)
            if len(x) != 0:
                x = self.vectors.query(x)
                x = _pad_zeros(x, self.MAX_SEQ_LEN)
            else:
                x = np.zeros((self.MAX_SEQ_LEN, self.vectors.dim))
            return x

        def _process_datapair(X, y):
            X = tf.py_function(_process_string, [X], tf.float32)
            X.set_shape([self.MAX_SEQ_LEN, self.vectors.dim])
            y.set_shape([])
            return X, y

        dataset = dataset.map(_process_datapair)
        return dataset.shuffle(buffer_size=1000).batch(self.BATCH_SIZE).prefetch(8)

    def get_dataset(self):
        self.read_data()
        return self.prepare_dataset()


# test
if __name__ == "__main__":
    tf.enable_eager_execution()
    wrapper = DataWrapperV2(PROCESSED_TRAIN, MAG_FILE)
    dataset = wrapper.get_dataset()
    for d in iter(dataset):
        print(d)
        break
