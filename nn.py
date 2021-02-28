import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras


np.random.seed(1)


# iris = pd.read_csv('data/iris.csv')
# X_ = iris.iloc[:, 0:4].values
# y_ = iris.iloc[:, -1].values
# y_ = LabelEncoder().fit_transform(y_)
# X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=.5, stratify=y_)


class NN:

    def __init__(self, n_epochs=50, eta=0.01, n_hidden=10, batch_size=10, seed=1):
        self.n_epochs = n_epochs
        self.eta = eta
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed=seed)
        self.b_h, self.W_h, self.b_o, self.W_o = None, None, None, None

    @staticmethod
    def onehot(y):
        n_unique = np.unique(y).shape[0]
        Y = np.zeros([n_unique, y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def forward(self, X):
        Z_h = self.b_h + np.dot(X, self.W_h)
        A_h = self.activation(Z_h)
        Z_o = self.b_o + np.dot(A_h, self.W_o)
        A_o = self.activation(Z_o)
        return Z_h, A_h, Z_o, A_o

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, l2=0.):
        Y = self.onehot(y)
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = self.random.normal(loc=0, scale=0.01, size=[X.shape[1], self.n_hidden])
        self.b_o = np.zeros(Y.shape[1])
        self.W_o = self.random.normal(loc=0, scale=0.01, size=[self.n_hidden, Y.shape[1]])
        for e in range(self.n_epochs):
            print('epoch %d' % e)
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx+self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = Y[batch] - A_o
                sigmoid_derivative = A_h * (1 - A_h)
                delta_h = np.dot(delta_o, self.W_o.T) * sigmoid_derivative
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0) + self.b_o * l2
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o) + self.W_o * l2
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0) + self.b_h * l2
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h) + self.W_h * l2
        return self


# perc = Perceptron(n_epochs=800, eta=0.001)
# perc.fit(X_train, y_train, l2=0.001)
# results = perc.predict(X_test)
# print(
#     np.sum(np.argmax(results, axis=1) == y_test) / y_test.shape[0]
# )


#   MNIST


# mnist = pd.read_csv('data/mnist.csv')
# X_ = mnist.iloc[:, 1:-1].values / 255.
# y_ = mnist.iloc[:, 0].values
# X_train, X_test, y_train, y_test = train_test_split(
#     X_, y_, train_size=0.5, stratify=y_
# )


mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255., X_test / 255.
X_train = np.array([image.reshape(-1) for image in X_train])
X_test = np.array([image.reshape(-1) for image in X_test])

nn = NN(n_epochs=20, batch_size=100, eta=0.01)
nn.fit(X_train, y_train)
results = nn.predict(X_test)
print(
    np.sum(np.argmax(results, axis=1) == y_test) / y_test.shape[0]
)




































































































































































































































































































