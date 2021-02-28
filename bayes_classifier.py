import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#  TODO single datapoint structure, redo into multipoint bayes classifier


np.random.seed(0)


iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, 0:4].values
y_ = iris.iloc[:, -1].values
y_ = LabelEncoder().fit_transform(y_)
X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=.5, stratify=y_)


class BayesClassifier:

    def __init__(self, X, y):
        self.m, self.v = self.means_and_vars(self.dictionary_of_samples(X, y))
        self.y_proportions = {}
        for c in np.unique(y):
            self.y_proportions[c] = np.sum(y == c) / y.shape[0]

    @staticmethod
    def dictionary_of_samples(X, y):
        d = {}
        for c in np.unique(y_):
            d[c] = []
        for i in range(X.shape[0]):
            d[y[i]].append(X[i])
        for k in d.keys():
            d[k] = np.array(d[k])
        return d

    @staticmethod
    def means_and_vars(d):
        means, vars = {}, {}
        for c in d.keys():
            means[c] = np.sum(d[c], axis=0) / d[c].shape[0]
            vars[c] = np.var(d[c], axis=0)
        return means, vars

    def proba(self, x, c):
        return np.sum(
            (1/np.sqrt(2*np.pi*self.v[c])) * np.exp(
                -(1/(2*self.v[c])) * (x - self.m[c]) ** 2
            )
        )

    def predict(self, x):
        values = []
        for c in self.m.keys():
            values.append(self.proba(x, c) + self.y_proportions[c])
        return values


bc = BayesClassifier(X_train, y_train)
print(
    np.argmax(bc.predict(X_test[2])) == y_test[2]
)
# print(
#     np.sum(bc.predict(X_))
# )



































































































































































































































































