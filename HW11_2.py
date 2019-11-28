from __future__ import print_function

import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt


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
        data[ind] = [int(x) for x in num[1:]]

    return data, labels


# gradient Descent

train_data, train_labels = read_data("sample_train.csv")
train_data = train_data / 255
cov_mat = np.cov(train_data.T)
alpha = 0.5
H = np.random.randn(784, 784)
Q, R = qr(H)
p = Q
print(p)
gradient = np.zeros((1, 784))
for i in range(100):
    for j in range(2):
        for k in range(3, 784):
            gradient += 6000*np.matmul(cov_mat, p[:, k])
        p[:, j] = p[:, j] - alpha * 2 * gradient
z = np.matmul(train_data, p[:, 0:2])
plt.plot(z, 'r+')
plt.show()
evalue, evct = np.linalg.eig(cov_mat)
s = (p[:, 0:2] - evct[:, 0:2])
plt.plot(s,'g*')
plt.show()
q = train_data.dot(evct[:, 0:2])
plt.plot(q[:, 0], q[:, 1], 'b+')
plt.title('Data plot for 2 D PCA')
plt.show()