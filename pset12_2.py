import numpy as np


def val(w, x):
    sum = 0
    for i in range(x.shape[0]):
        sum += w[i] * x[i]
    if sum >= 0:
        return 1
    else:
        return -1


def Perceptron(x, y, w):
    wOld = w
    wNew = 0
    n = 1
    itr = 100
    for i in range(itr):
        sum = np.zeros(x.shape[1])
        for j in range(x.shape[0]):
            if y[j] * val(wOld, x[j]) < 0:
                sum += x[j] * y[j]
                # print(sum)
        wNew = wOld + n * sum
        # Convergence Criteria is satisfied or no of iteration < itr
        if np.array_equal(wOld, wNew):
            break
        wOld = wNew
        print(wNew)
    print('Final Value of W')
    print(wNew)


x = np.array([[1, 1, 1],
              [-1, -1, 1],
              [2, 2, 1],
              [-2, -2, 1],
              [-1, 1, 1],
              [1, -1, 1]])
y = np.array([[-1],
              [-1],
              [1],
              [-1],
              [1],
              [1]])
wOld = [0, 1, -1]
Perceptron(x, y, wOld)
# For the 1st Question it converges
# For 2nd Question it oscillates