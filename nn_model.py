
import numpy as np
import tensorflow as tf

from model import Model


class NNModel(Model):
    def __init__(self, hidden_units: int = 256):
        self.hidden_units = hidden_units
        self.input_units = None
        self.output_units = None
        self.best_hyper_parameters = None
        self.tuner = None
        self._is_tuned = False

    def train(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, epochs=50
    ):
        self.input_units = x.shape[1]
        self.output_units = y.shape[1]
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.input_units,)),
                tf.keras.layers.Dense(self.hidden_units, activation="relu"),
                tf.keras.layers.Dense(self.output_units, activation="softmax"),
            ]
        )
        self.model = model
        loss = "categorical_crossentropy"
        metrics = ["mae"]
        model.compile(optimizer="adam", loss=loss, metrics=metrics)

        model.summary()
        metrics = ["mae"]
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, mode="min", restore_best_weights=True
        )
        model.fit(x, y, validation_data=(vx, vy), epochs=epochs, callbacks=[callback])

    def eval(self, x, y):
        results = self.model.evaluate(x, y, batch_size=128)
        print(results)

    def predict(self, x: np.ndarray):
        return self.model.predict(x)

    def save(self, path: str):
        model_dir = path + "/model"
        self.model.save(model_dir)

    def load(self, path: str):
        model_dir = path + "/model"
        self.model = tf.keras.models.load_model(model_dir)
