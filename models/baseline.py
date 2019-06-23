import tensorflow as tf
import os
import sys

# fixes import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dataset.prepare_data import DataWrapper
from dataset.process_raw import PROCESSED_TRAIN, PROCESSED_VAL
from mag_model.mag_model import MAG_FILE


class Model:
    def __init__(self, datafile_path, mag_file_path, batch_size=32, max_seq_len=200):
        self.data_wrapper = DataWrapper(
            datafile_path, mag_file_path, batch_size, max_seq_len
        )

    def build_model(self):
        i = tf.keras.layers.Input(
            shape=(self.data_wrapper.MAX_SEQ_LEN, self.data_wrapper.vectors.dim)
        )
        Bidir_LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, activation="tanh", return_sequences=True),
            merge_mode="concat",
        )
        Bidir_LSTM_out = Bidir_LSTM(i)
        maxpool = tf.keras.layers.GlobalMaxPooling1D()(Bidir_LSTM_out)
        hidden = tf.keras.layers.Dense(512)(maxpool)
        output = tf.keras.layers.Dense(1, activation="softmax")(hidden)
        model = tf.keras.Model(inputs=i, outputs=output)
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
        self.model = model


# testing
if __name__ == "__main__":
    model = Model(PROCESSED_TRAIN, MAG_FILE)
    model.build_model()
