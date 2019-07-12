import tensorflow as tf
import os
import sys
import math
from dataset.prepare_data import DataWrapperV2
from dataset.process_raw import PROCESSED_TRAIN, PROCESSED_VAL
from mag_model.mag_model import MAG_FILE

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(DIR, "model.h5")


class Model:
    def __init__(
        self,
        train_datafile_path,
        val_datafile_path,
        test_datafile_path,
        mag_file_path,
        batch_size=32,
        max_seq_len=100,
        epochs=2,
    ):
        # training data
        self.train_data_wrapper = DataWrapperV2(
            train_datafile_path, mag_file_path, batch_size, max_seq_len
        )
        # validation data
        self.val_data_wrapper = DataWrapperV2(
            val_datafile_path, mag_file_path, batch_size, max_seq_len
        )
        # test data
        self.test_data_wrapper = DataWrapperV2(
            test_datafile_path, mag_file_path, batch_size, max_seq_len
        )
        self.model = None
        self.epochs = epochs

    def build_model(self):
        print("building model")
        i = tf.keras.layers.Input(
            shape=(
                self.train_data_wrapper.MAX_SEQ_LEN,
                self.train_data_wrapper.vectors.dim,
            )
        )
        Bidir_LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True),
            merge_mode="concat",
        )
        Bidir_LSTM_out = Bidir_LSTM(i)
        maxpool = tf.keras.layers.GlobalMaxPooling1D()(Bidir_LSTM_out)
        hidden = tf.keras.layers.Dense(128)(maxpool)
        output = tf.keras.layers.Dense(1, activation="softmax")(hidden)
        model = tf.keras.Model(inputs=i, outputs=output)
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
        self.model = model

    def train(self):
        # if the model is not build
        if self.model == None:
            self.build_model()
        else:
            print("fitting model")
            self.model.fit(
                self.train_data_wrapper.get_dataset(),
                validation_data=self.val_data_wrapper.get_dataset(training=False),
                steps_per_epoch=int(
                    math.floor(
                        len(self.train_data_wrapper.X)
                        / self.train_data_wrapper.BATCH_SIZE
                    )
                ),
                validation_steps=int(
                    math.floor(
                        len(self.val_data_wrapper.X) / self.val_data_wrapper.BATCH_SIZE
                    )
                ),
                epochs=self.epochs,
            )
        self.model.save(MODEL_FILE)

    def evaluate(self):
        if self.model == None:
            self.train()
        else:
            print("evaluating model on test set")
            testdata = self.test_data_wrapper.get_dataset(training=False)
            self.model.evaluate(testdata, verbose=0)


# testing
if __name__ == "__main__":
    model = Model(PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_VAL, MAG_FILE)
    model.build_model()
    model.train()
