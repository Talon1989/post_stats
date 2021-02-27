import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


np.random.seed(1)


iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, 0:4].values
y_ = iris.iloc[:, -1].values
y_ = LabelEncoder().fit_transform(y_)
X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=.5, stratify=y_)


class Perceptron:

    def __init__(self, n_epochs=50, eta=0.01, batch_size=10, seed=1):
        self.n_epochs = n_epochs
        self.eta = eta
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed=seed)
        self.b, self.W = None, None

    @staticmethod
    def onehot(y):
        n_unique = np.unique(y).shape[0]
        Y = np.zeros([n_unique, y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    def net_input(self, X):
        return self.b + np.dot(X, self.W)

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=+250)))

    def predict(self, X):
        return self.net_input(X)

    def fit(self, X, y, l2=0.):
        Y = self.onehot(y)
        self.b = np.zeros(Y.shape[1])
        self.W = np.zeros([X.shape[1], Y.shape[1]])
        for e in range(self.n_epochs):
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)
            square_error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx+self.batch_size]
                error = Y[batch] - self.activation(self.net_input(X[batch]))
                db = - np.sum(error, axis=0)
                dW = - np.dot(X[batch].T, error)
                self.b = self.b - self.eta * db + l2*self.b
                self.W = self.W - self.eta * dW + l2*self.W
                square_error += np.sum(np.sum(error**2))
            print('epoch %d, square error: %f' % (e, square_error))
        return self


perc = Perceptron(n_epochs=500, eta=0.005)
perc.fit(X_train, y_train, l2=0.01)
results = perc.predict(X_test)
print(
    np.sum(np.argmax(results, axis=1) == y_test) / y_test.shape[0]
)








































































































































































































































































































