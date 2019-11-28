import numpy as np


def cost(y, t):
    h = -np.multiply(y, np.log(t))
    s = 0
    for i in range(h.shape[0]):
        s += np.sum(h[i])
    s = s / h.shape[0]
    return s


def softmax_activation(x, w):
    z = np.dot(x, w)
    t = np.zeros((z.shape[0], z.shape[1]))
    for i in range(z.shape[0]):
        sum = np.sum(np.exp(z[i]))
     
        for j in range(z.shape[1]):
            t[i, j] = np.exp(z[i, j]) / sum
    return t


def gradient_cal(x, y, o):
    diff = y - o
    j = np.mat(x.T) * np.mat(diff)
    p = 0
    for i in range(x.shape[1]):
        p = p + j[i]
    p = p / x.shape[0]
  
    return - np.sum(p)


def gradient(x, y, w):
    wOld = w
    wNew = 0
    n = .1
    itr = 100
    c = 0
    while c < itr:
        o = softmax_activation(x, wOld)
        # print(o.shape)
        wNew = wOld - n * gradient_cal(x, y, o)
        if np.array_equal(wOld, wNew):
            break;
        print(cost(y, o))
        wOld = wNew
        c = c + 1


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


def scale(X, x_min, x_max):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


train_data, train_labels = read_data("sample_train.csv")
k = 10

x = np.array(train_data)
x = scale(x, -1, 1)

print(np.max(x))
y = np.array(train_labels)
y1 = np.zeros((x.shape[0], k))

for i in range(x.shape[0]):
    y1[i, np.int(y[i])] = 1
y = y1
# # dimension of x: n*m, n= no. of samples, m= features
w = np.random.normal(0, 5, (x.shape[1] + 1, k))

# # dimension of w: (m+1)Xk, k is no of classes
x1 = np.ones((x.shape[0], 1))
x = np.append(x1, x, axis=1)
gradient(x, y, w)
# t = softmax_activation(x, w)
# print(t)