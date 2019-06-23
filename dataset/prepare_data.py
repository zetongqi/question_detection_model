import os
import sys
import tensorflow as tf
from dataset.process_raw import PROCESSED_TRAIN, PROCESSED_VAL
import csv
from nltk.tokenize import word_tokenize

# fixes import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mag_model.mag_model import MAG_FILE
from pymagnitude import *
import numpy as np


class DataWrapper:
    def __init__(self, datafile_path, mag_file_path, batch_size=32, max_seq_len=200):
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

    # generator for tf.data.Dataset object
    def data_generator(self):
        def pad_zeros(x, max_seq_len):
            if len(x) >= max_seq_len:
                return x[0:max_seq_len, :]
            else:
                return np.concatenate(
                    (x, np.zeros(max_seq_len - len(x), vectors.dim)), axis=0
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


# test
if __name__ == "__main__":
    wrapper = DataWrapper(PROCESSED_TRAIN, MAG_FILE)
    wrapper.read_data()
    print(type(wrapper.get_dataset()))
