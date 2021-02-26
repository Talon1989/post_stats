import numpy as np
import matplotlib.pyplot as plt
import plotly


np.random.seed(1)


arr = np.random.normal(10, 2, 10)
# arr_2 = 3*arr
arr_2 = np.random.normal(15, 1, 10)


# plt.scatter(arr, arr_2, c='b', s=2)
# plt.show()
# plt.clf()

D = np.hstack([
    np.random.normal(10, 2, 10).reshape(-1, 1),
    np.random.normal(15, 3, 10).reshape(-1, 1)
])


def projection_vector(A, b):
    x = np.dot(
        np.linalg.inv(np.dot(A.T, A)),
        np.dot(A.T, b)
    )
    return np.dot(A, x)


aug_arr = np.hstack([np.ones(10).reshape(-1, 1), arr.reshape(-1, 1)])
p = projection_vector(aug_arr, arr_2.reshape(-1, 1))

plt.axis('equal')
plt.scatter(arr, arr_2)
plt.plot(arr, p, c='r')
plt.show()
plt.clf()










































