import tensorflow as tf
import keras
import os
import sys
import math
from dataset.process_raw import PROCESSED_TRAIN, PROCESSED_VAL
from mag_model.mag_model import MAG_FILE
from models.get_bert import BERT_DIR, CHECKPOINT_FILE, CONFIG_JSON, VOCAB_FILE
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.utils import to_categorical
import json
import tokenization
import csv
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(DIR, "model.h5")


def bert_tokenize(input_str, max_seq_len=512):
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
    tokens_temp = tokenizer.tokenize(input_str)
    tokens = []
    tokens.append("[CLS]")
    tokens.extend(tokens_temp)
    paddings = ["[PAD]"] * (max_seq_len - len(tokens_temp) - 2)
    tokens.extend(paddings)
    tokens.append("[SEP]")
    token_index = tokenizer.convert_tokens_to_ids(tokens)
    seg_id = [0] * len(tokens)
    return (token_index, seg_id)


# reads the processed data and return data matrices
def read_data(datafile_path):
    X = []
    y = []
    tsvFile = csv.reader(open(datafile_path), delimiter="\t")
    for row in tsvFile:
        X.append(row[0])
        y.append(int(row[1] == "1"))
    return X, y


class Model:
    def __init__(self, max_seq_len=128, batch_size=16):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def build_model(self):
        basemodel = load_trained_model_from_checkpoint(CONFIG_JSON, CHECKPOINT_FILE)
        x = basemodel(basemodel.inputs)
        avg_pooling = keras.layers.GlobalAveragePooling1D()(x)
        dropout = keras.layers.Dropout(0.5)(avg_pooling)
        out = keras.layers.Dense(2, activation="softmax")(dropout)
        model = keras.Model(inputs=basemodel.inputs, outputs=out)
        model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["acc"])
        model.summary()
        self.model = model

    def train(self, TRAIN_FILE):
        if self.model == None:
            self.build_model()


# testing
if __name__ == "__main__":
    bert = Model()
    bert.build_model()
