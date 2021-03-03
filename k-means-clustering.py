from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(1)

blobs = datasets.make_blobs(500, 2)
X_ = blobs[0]
y_ = blobs[1]


# def distance(X, indices):
#     distances = []
#     for idx in indices:
#         distances.append(
#             np.sqrt(np.sum((X[idx] - X) ** 2, axis=1))
#         )
#     return np.array(distances)


def distance(X, points):
    distances = []
    for point in points:
        distances.append(
            np.sqrt(np.sum((point - X) ** 2, axis=1))
        )
    return np.array(distances)


n_predicted_clusters = 3


#
#
# distances = distance(X_, X_[pivots_indices])
#
#
# points_in_clusters = [[] for _ in range(n_predicted_clusters)]
# for i in range(distances[0].shape[0]):
#     min_distance_idx = int(np.argmin(distances[:, i]))
#     points_in_clusters[min_distance_idx].append(X_[i])
# for c in range(n_predicted_clusters):
#     points_in_clusters[c] = np.array(points_in_clusters[c])
# new_pivots = [
#     np.sum(points_in_clusters[c], axis=0) / points_in_clusters[c].shape[0] for c in range(n_predicted_clusters)
# ]


# print(new_pivots)


def calculate_pivots(X, current_pivots):
    points_in_clusters = [[] for _ in range(n_predicted_clusters)]
    distances = distance(X, current_pivots)
    for i in range(distances[0].shape[0]):
        min_distance_idx = int(np.argmin(distances[:, i]))
        points_in_clusters[min_distance_idx].append(X[i])
    for c in range(n_predicted_clusters):
        points_in_clusters[c] = np.array(points_in_clusters[c])
    new_pivots = [
        np.sum(points_in_clusters[c], axis=0) / points_in_clusters[c].shape[0]
        for c in range(n_predicted_clusters)
    ]
    flag = True
    for c in range(n_predicted_clusters):
        if np.all(current_pivots[c] != new_pivots[c]):
            flag = False
            break
    if flag is True:
        return new_pivots, points_in_clusters
    else:
        return calculate_pivots(X, new_pivots)


pivots_indices = [np.random.randint(X_.shape[0]) for _ in range(n_predicted_clusters)]
clusters_pivots, clusters_points = calculate_pivots(X_, X_[pivots_indices])
clusters_pivots = np.array(clusters_pivots)

# plt.scatter(X_[:, 0], X_[:, 1], c=y_)
for c in range(len(clusters_points)):
    plt.scatter(clusters_points[c][:, 0], clusters_points[c][:, 1])
plt.scatter(clusters_pivots[:, 0], clusters_pivots[:, 1], c='r')
plt.show()
plt.clf()














































































































































































































