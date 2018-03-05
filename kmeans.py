import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.cluster as cluster


def k_means(X, k, max_inter=300):
    dataset = np.zeros((X.shape[0], X.shape[1]+1))
    dataset[:, :-1] = X
    centroids = np.zeros((k, X.shape[1]))
    for i, v in enumerate(random.sample(range(X.shape[0]), k=k)):  # choose randomly k centroids
        centroids[i] = X[v]
    while max_inter:
        dataset = get_dataset(dataset, centroids)  # get current iteration dataset with labels
        centroids = get_centroids(dataset, centroids)  # update centroids according to the current dataset
        max_inter -= 1
    inertia = get_inertia(dataset, centroids)  # get the final value of the inertia criterion
    return centroids, dataset[:, -1], inertia


def get_dataset(dataset, centroids):
    distances = np.zeros((dataset.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        # Euclidean distance, calculate the distance from each centroid to all samples
        distances[:, i] = np.sum((dataset[:, :-1] - centroid)**2, axis=1)**.5
    dataset[:, -1] = np.argmin(distances, axis=1)  # make a label for each sample
    return dataset


def get_centroids(dataset, centroids):
    centroids = np.zeros((centroids.shape[0], centroids.shape[1]))
    n_counts = [0]*centroids.shape[0]  # the number of each clustering(the current iteration)
    for sample in dataset:
        centroids[int(sample[-1])] += sample[:-1]
        n_counts[int(sample[-1])] += 1
    for i, n_count in enumerate(n_counts):
        centroids[i] /= n_count  # update centroids
    return centroids


def get_inertia(dataset, centroids):
    sum_of_squares = 0.
    for sample in dataset:
        sum_of_squares += euclidean_distance(sample[:-1], centroids[int(sample[-1])])**2
    return sum_of_squares


def euclidean_distance(x, y):
    return np.sum((x - y)**2)**.5  # 2-norm


def converter1(label):
    if label == b'Iris-setosa':
        return 0
    elif label == b'Iris-versicolor':
        return 1
    else:
        return 2


def test_data():
    l = [[1, 1, 0], [2, 1, 0], [1, 2, 0], [10, 10, 1], [8, 9, 1], [11, 9, 1], [-4, -6, 2], [-5, -4, 2]]
    # l = [[0, 1, 0], [2, 1, 0], [10, 10, 1], [8, 12, 1]]
    return np.array(l)


def plot_clusterings(dataset):
    pca_dataset = np.zeros((dataset.shape[0], 3))
    pca_dataset[:, -1] = dataset[:, -1]
    pca = PCA(n_components=2)
    pca_dataset[:, :-1] = pca.fit_transform(dataset[:, :-1])
    colors = ['red', 'green', 'blue']
    plt.figure(num='MLee')
    for i in range(len(colors)):
        x0 = pca_dataset[:, 0][pca_dataset[:, -1] == i]
        x1 = pca_dataset[:, 1][pca_dataset[:, -1].astype('int') == i]
        plt.scatter(x0, x1, c=colors[i])
    plt.title('Iris Data Set')
    plt.show()


if __name__ == '__main__':
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dataset = np.loadtxt(url, delimiter=',', converters={4: converter1})
    # dataset = test_data()
    X, _ = dataset[:, :-1], dataset[:, -1]
    centroids, label, inertia = k_means(X, 3)
    print((centroids, label, inertia))
    # print(cluster.k_means(X, 3))
    plot_clusterings(np.concatenate((X, label.reshape(-1, 1)), axis=1))
    # plot_clusterings(dataset)
