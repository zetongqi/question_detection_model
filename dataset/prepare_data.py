import os
import sys
import tensorflow as tf
from process_raw import PROCESSED_TRAIN, PROCESSED_VAL
import csv
from nltk.tokenize import word_tokenize
# fixes import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mag_model.mag_model import MAG_FILE
from pymagnitude import *


# returns magnitude object
def get_mag_model(MAG_PATH):
	return Magnitude(MAG_PATH)


# reads the processed data and return data matrices
def data_reader(FILE_PATH):
	X = []
	y = []
	tsvFile = csv.reader(open(FILE_PATH), delimiter='\t')
	for row in tsvFile:
		X.append(row[0])
		y.append(row[1])
	return (X, y)


# tokenization method
def tokenize(sequence):
	return word_tokenize(sequence)


# generator for tf.data.Dataset object
def data_generator(X, y):
	vectors = get_mag_model(MAG_FILE)

	def pad_zeros()

	for x, y in zip(X, y):
		yield vectors.query(tokenize(x)), y


# returns tf.data.Dataset object for model.fit
def prepare_data(generator):
	tf.data.Dataset.from_generator(generator, (tf.float32, tf.uint8), (tf.TensorShape([]), tf.TensorShape([None])))