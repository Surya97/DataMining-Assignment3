import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def find_number_of_clusters(x):
    x = np.array(x)
    plt.figure(figsize=(10, 7))
    plt.title("Carbohydrate value clustering")
    dend = shc.dendrogram(shc.linkage(x, method='ward'))
    plt.show()


def compute(x):
    x = np.array(x)
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    print(cluster.fit_predict(x))
    plt.figure(figsize=(10, 7))
    plt.scatter(x[:, 0], x[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()
