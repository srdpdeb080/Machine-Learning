import numpy as np
from sklearn.linear_model import LogisticRegression
import operator


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


train_data, train_labels = read_data("sample_train.csv")
x = train_data
y = train_labels
k = 10
x1 = np.ones((x.shape[0], 1))
x = np.append(x, x1, axis=1)
index = np.arange(0, 5999, 600)
x_new = np.zeros((x.shape[0], x.shape[1]))
y_new = np.zeros(y.shape[0])
for i in range(x.shape[0]):
    ind = index[np.int(y[i])]
    index[np.int(y[i])] += 1
    x_new[ind] = x[i]
    y_new[ind] = y[i]

classifier = np.zeros((10, 10, 785))
for i in range(k - 1):
    x1 = x_new[i * 600: (i * 600) + 600, :]
    y1 = y_new[i * 600: (i * 600) + 600]
    for j in range(i + 1, k):
        x2 = x_new[j * 600: (j * 600) + 600, :]
        y2 = y_new[j * 600: (j * 600) + 600]
        x_data = np.append(x1, x2, axis=0)
        y_data = np.append(y1, y2, axis=0)
      
        lr = LogisticRegression()
        lr.fit(x_data, y_data)
        c = lr.coef_
        classifier[i, j] = c[0]
        classifier[j, i] = c[0]

test_data, test_labels = read_data("sample_test.csv")
x1 = np.ones((test_data.shape[0], 1))
x = np.append(test_data, x1, axis=1)
y_test = np.zeros(test_labels.shape[0])

for i in range(test_data.shape[0]):
    major = np.zeros(k)
    for m in range(k):
        for n in range(m + 1, k):
            d = np.sum(np.multiply(x[i], classifier[m, n]))
            if d >= 0:
                major[m] = major[m] + 1
                # print(m)
            else:
                major[n] = major[n] + 1
                # print(n)
    max_index, max_value = max(enumerate(major), key=operator.itemgetter(1))
    y_test[i] = max_index
c = 0
for i in range(test_labels.shape[0]):
    if y_test[i] == test_labels[0]:
        c += 1
print(c / 10)