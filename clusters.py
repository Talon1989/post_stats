import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(1)

blobs = datasets.make_blobs(500, 2)
X = blobs[0]
y = blobs[1]

a_X = blobs[0][y != 2]
a_y = blobs[1][y != 2]

# plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], s=2, c='b')
# plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], s=2, c='r')

plt.scatter(X[:, 0], X[:, 1], s=2, c=y)

# plt.scatter(a_X[:, 0], a_X[:, 1], s=2, c=a_y)

plt.show()
plt.clf()


