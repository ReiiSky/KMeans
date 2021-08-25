from typing import List
from math import sqrt
from numpy import array, append, sum, copy

# Dataset
# [?][0] = gender - 0 = man - 1 = woman
# [?][1] = age
# [?][2] = symptom
# [?][3] = suffering duration
sample_data = [
    [0, 67, 13, 5],
    [0, 60, 15, 3],
    [1, 65, 16, 8],
    [0, 62, 14, 4],
    [1, 69, 15, 3],
    [1, 70, 11, 4],
    [0, 66, 16, 2],
    [1, 64, 12, 5],
    [0, 65, 14, 5],
    [1, 69, 19, 7],
    [0, 71, 13, 6],
    [1, 62, 11, 5],
    [1, 64, 15, 5],
    [0, 63, 12, 6],
    [0, 60, 14, 7],
    [1, 66, 11, 5],
    [0, 67, 19, 9],
    [1, 61, 13, 4],
    [1, 74, 12, 3],
    [0, 72, 15, 6],
]

dataset = array(sample_data)[:, 1:]
initial_cluster_point = array([
    dataset[6],
    dataset[12],
], dtype=float)


class KMeans:
    def __init__(self, initial_cluster):
        self.cluster_matrix = initial_cluster

    def f_distance(self, array_1: List[int], array_2: List[int]):
        if(len(array_1) != len(array_2)):
            print("Length of array_1 not same as length array_2")
        
        sum_of_diff = 0
        for index in range(len(array_1)):
            sum_of_diff += (array_1[index] - array_2[index] ) ** 2
        return sqrt(sum_of_diff)
    
    def fit(self, dataset, iteration_threshold = 1):
        if iteration_threshold < 0:
            print("iteration_threshold should larger than 0")
            exit()

        dataset_len_column = dataset.shape[1]
        last_distance_threshold = float("inf")
        while last_distance_threshold > iteration_threshold:
            iteration_matrix = array([], dtype=float)
            for i in range(len(dataset)):

                last_distance_to_data = float("inf")
                iteration_matrix = append(iteration_matrix, 0)

                for j in range(len(self.cluster_matrix)):
                    distance_to_data = self.f_distance(
                        self.cluster_matrix[j],
                        dataset[i],
                    )
                    if distance_to_data < last_distance_to_data:
                        last_distance_to_data = distance_to_data
                        iteration_matrix[i] = j

            new_cluster_matrix = copy(self.cluster_matrix)
            for i in range(len(new_cluster_matrix)):
                for j in range(dataset_len_column):
                    classified_data = dataset[:, j][iteration_matrix == i]
                    new_value_point_matrix = sum(classified_data) / (len(classified_data))
                    new_cluster_matrix[i][j] = new_value_point_matrix
            last_distance_threshold = sum((self.cluster_matrix - new_cluster_matrix) ** 2) / self.cluster_matrix.size
            self.cluster_matrix = new_cluster_matrix
    
    def predict(self, unpredicted_data):
        m = []
        for i in range(len(unpredicted_data)):
            last_distance_to_data = float("inf")
            m.append(0)
            for j in range(len(self.cluster_matrix)):
                d = self.f_distance(
                    self.cluster_matrix[j],
                    unpredicted_data[i],
                )
                if d < last_distance_to_data:
                    m[i] = j
                    last_distance_to_data = d
        return array(m)
    
classifier = KMeans(initial_cluster_point)
classifier.fit(dataset, 0)
p = classifier.predict(dataset)
# print(p)
# from sklearn.cluster import KMeans
# classifier = KMeans(n_clusters=2, random_state=0)
# classifier = classifier.fit(dataset)
# p = classifier.predict(dataset)
# print(p)
# print(classifier.cluster_centers_)

max_each_column = [max(dataset[:, i]) for i in range(dataset.shape[1]) ]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    dataset[:, 0][p == 0]/max_each_column[0],
    dataset[:, 1][p == 0]/max_each_column[1],
    dataset[:, 2][p == 0]/max_each_column[2],
    c="red")

ax.scatter(
    classifier.cluster_matrix[0][0]/max_each_column[0],
    classifier.cluster_matrix[0][1]/max_each_column[1],
    classifier.cluster_matrix[0][2]/max_each_column[2],
    marker="x",
    c="red")

ax.scatter(
    dataset[:, 0][p == 1]/max_each_column[0],
    dataset[:, 1][p == 1]/max_each_column[1],
    dataset[:, 2][p == 1]/max_each_column[2],
    c="blue")

ax.scatter(
    classifier.cluster_matrix[1][0]/max_each_column[0],
    classifier.cluster_matrix[1][1]/max_each_column[1],
    classifier.cluster_matrix[1][2]/max_each_column[2],
    marker="x",
    c="blue")

plt.show()