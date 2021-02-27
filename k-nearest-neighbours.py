import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#  TODO single datapoint structure, redo into multipoint array classificator


np.random.seed(1)

iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, 0:4].values
y_ = iris.iloc[:, -1].values


class K_nearest:

    def __init__(self, k=5):
        self.k = k

    @staticmethod
    def distance(arr_1: np.array, matrix: np.array):
        assert arr_1.shape[0] == matrix.shape[1]
        return np.sum((arr_1 - matrix) ** 2, axis=1)

    def fit(self, datapoint: np.array, dataset: np.array):
        """
        :param datapoint: sample
        :param dataset: matrix of datapoints
        :return: closest k indices
        """
        distance_array = self.distance(datapoint, dataset)
        return distance_array.argsort()[:self.k]


datapoint_ = X_[84]
k = K_nearest()
closest = k.fit(datapoint_, X_)
print(y_[closest])


































































































































































