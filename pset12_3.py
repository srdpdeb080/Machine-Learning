import numpy as np
import pandas as pd


def cost(x, w, y):
    err = np.dot(x, w) - y
    j = np.sum(err ** 2)*(1/x.shape[0])
    return j


def gradient_descent(x, y, n):
    wOld = np.random.random_integers(0, 10, (x.shape[1], 1))
    itr = 1000
    c = 0
    while (c < itr):
        sum = np.dot((np.dot(x, wOld) - y).T, x)
        sum = sum.T
        wNew = wOld - n * (1 / x.shape[0]) * sum
        c += 1
        if np.array_equal(wOld, wNew):
            break
        wOld = wNew
        err = cost(x, wNew, y)
        print(err)
    print("no of steps: ",c)


data = np.array(pd.read_csv("wine.data"))
x = data[:, 0:13]
y = data[:, 13:14]
gradient_descent(x, y, 0.01)


# calculating optimal leaning rate
wOld = np.random.random_integers(0, 10, (x.shape[1], 1))
delJ = np.dot(wOld.T,wOld)
hessian = np.dot(x.T, x)
a = np.dot(wOld.T,hessian)
a = np.dot(a,wOld)
n = delJ[0, 0]/a[0, 0]
print('Gradient descent using optimal learning Rate')
gradient_descent(x, y, n)
print(n)