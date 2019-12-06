import read_data
import matplotlib.pyplot as plt
import numpy as np
import k_means
import dbscan
import agglomerative


df = read_data.read_csv('./mealAmountData/mealAmountData3.csv')
print(df)
print(df.shape)
df.info()
print(df.describe())
xs = [x for x in range(df.shape[0])]
ys = df.iloc[:, 0]
# plt.scatter(xs, ys)
# plt.show()
matx = []
for i in range(len(xs)):
    temp = list()
    temp.append(i)
    temp.append(ys[i])
    matx.append(temp)

k_means.show_clusters(matx)
# dbscan.compute(matx)
agglomerative.find_number_of_clusters(matx) #-> finds the number of clusters to consider by plotting the dendrogram
agglomerative.compute(matx)