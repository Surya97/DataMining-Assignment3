from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def compute(x):
    x = np.array(x)
    db = DBSCAN(eps=40, min_samples=3)
    db.fit(x)
    y_pred = db.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='Paired')
    plt.title("DBSCAN")
    plt.show()