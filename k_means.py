from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fit(x):
    x = np.array(x)
    for k in range(1, 10):
        sse = {}
        for k in range(1, 10):
            x = pd.DataFrame(x)
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)
            x["clusters"] = kmeans.labels_
            # print(data["clusters"])
            sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()


def show_clusters(x):
    x = np.array(x)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(x)
    pred_kmeans = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=pred_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

