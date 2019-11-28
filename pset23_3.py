from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
   
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
   
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
       
    return (data, labels)


train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
pca = PCA(n_components=2)
pca.fit(test_data)
test = pca.transform(test_data)
pca.fit(train_data)
train = pca.transform(train_data)
print(test_data.shape)
print(train.shape)
kmeans = KMeans(n_clusters=10, random_state=0).fit(train)


predicted = kmeans.predict(test)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
scatter = ax.scatter(test[:,0],test[:,1],c=test_labels)
ax.legend(*scatter.legend_elements())
plt.show()




fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
scatter = ax.scatter(test[:,0],test[:,1],c=predicted)
ax.legend(*scatter.legend_elements())
plt.show()