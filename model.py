
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, InputLayer

from minisom import MiniSom

import pickle

import logging
logger = logging.getLogger(__name__)


class Model:
    MS_SHAPE = 10, 10
    MS_SIGMA = 1.0
    MS_LEARNING_RATE = .5

    DNN_HIDDEN_LAYERS = [128, 64, 32]
    DNN_LEARNING_RATE = .05
    DNN_L2_LAMBDA = .01
    DNN_EPOCHS = 1000

    def __init__(self, datapath: str):
        self._datapath = datapath
        df = pd.read_csv(self._datapath)
       

        X = df.iloc[:, :7].values  # Testing X, we need to limit this
        self.X = (X - np.min(X, axis=0)) / \
            (np.max(X, axis=0) - np.min(X, axis=0))

        self.labels, self.Y = np.unique(
            [" / ".join(_) for _ in df.iloc[:, [7, 9]].to_numpy()], return_inverse=True)

        assert len(self.X) == len(self.Y)

        self.ms = None
        self.dnn = None

    def generate(self):
        X = self.X
        Y = self.Y

        # MiniSOM
        self.ms = ms = MiniSom(
            x=self.MS_SHAPE[0],
            y=self.MS_SHAPE[1],
            input_len=X.shape[1],
            sigma=self.MS_SIGMA,
            learning_rate=self.MS_LEARNING_RATE
        )

        ms.random_weights_init(X)
        ms.train(
            data=X,
            num_iteration=100,
            verbose=True,
            random_order=True
        )

        logger.debug(f"Weights: {ms._weights.shape}")

        ws = np.array([ms.winner(_) for _ in X])

        # DNN
        dnn_input_size = 2
        dnn_output_size = len(np.unique(Y))

        self.dnn = dnn = Sequential()
        dnn.add(InputLayer(input_shape=(dnn_input_size, )))

        for sz in self.DNN_HIDDEN_LAYERS:
            dnn.add(Dense(sz, activation="relu"))
            # kernel_regularizer = regularizers.l2(DNN_L2_LAMBDA)

        dnn.add(Dense(dnn_output_size, activation="softmax"))
        dnn.compile(loss="sparse_categorical_crossentropy",
                    optimizer="adam", metrics=["accuracy"])

        dnn.fit(ws, Y, epochs=self.DNN_EPOCHS, batch_size=32)
        dnn.summary()

        return ms, dnn

    def predict(self, X):
        return self.dnn.predict([tuple(int(_) for _ in self.ms.winner(X))])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self.ms, self.dnn), f)

    def use(self, path: str):
        with open(path, 'rb') as f:
            self.ms, self.dnn = pickle.load(f)


def save(o: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(o, f)

def use(path: str) -> Model:
    with open(path, 'rb') as f:
        return pickle.load(f)
