import numpy as np


class Model(object):
    def tune(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray):
        raise NotImplementedError

    def train(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray):
        raise NotImplementedError

    def set_partitioner(self, partitioner):
        raise NotImplementedError

    def need_partitioner(self):
        return False

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
