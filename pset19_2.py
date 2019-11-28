import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    p = 1 / (1 + np.exp(-x))
    return p


def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def MSEerr(n, yOut, y):
    err = (1 / n) * np.sum((y - yOut) ** 2)
    return err


def hinge_loss(n, yOut, y):
    err = (1 / n) * np.sum(np.maximum(0, (1 - (y * yOut))))
    print(err)
    return err


def backword(y, x, x1, x2, x3, x4, w1, w2, w3, w4):
    param = 0.1
    w4prime = np.dot(w4, np.dot(sigmoidPrime(x4).T, (y - x4)))
    # new value of w4
    w4 = w4 - param * w4prime
    err3 = np.dot(x2.T, np.dot(sigmoidPrime(x3), np.dot((y - x4).T, sigmoidPrime(x3[:, 1:4])) * w4))
    w3 = w3 - param * err3
    w2Prime = np.dot(sigmoidPrime(x1).T, x2[:, 1:6])
    w2 = w2 - param * w2Prime
    w1Prime = np.dot(sigmoidPrime(x).T, x1[:, 1:4])
    w1 = w1 - param * w1Prime
    return w1, w2, w3, w4


def forward(n, x, y, w1, w2, w3, w4):
    b = np.ones((n, 1))
    x1 = np.dot(x, w1)
    # output of the first hidden layer
    x1 = sigmoid(x1)
    # for bias
    x1 = np.append(b, x1, axis=1)
    x2 = np.dot(x1, w2)
    # output of the 2nd hidden layer
    x2 = sigmoid(x2)
    x2 = np.append(b, x2, axis=1)
    # output of the 3rd hidden layer
    x3 = sigmoid(np.dot(x2, w3))
    x3 = np.append(b, x3, axis=1)
    # final Output
    x4 = sigmoid(np.dot(x3, w4))
    return x1, x2, x3, x4


n = 8
x = np.random.random_integers(-5, 5, (n, 2))
y = np.random.random_integers(0, 1, (n, 1))
b = np.ones((n, 1))
x = np.append(b, x, axis=1)
# Weights for first hidden layer including bias
w1 = np.random.random_integers(-5, 5, (3, 3))
# weights for second hidden layer including bias
w2 = np.random.random_integers(-5, 5, (4, 5))
# weights for third hidden layer including bias
w3 = np.random.random_integers(-5, 5, (6, 3))
# weights for 4th hidden layer including bias
w4 = np.random.random_integers(-5, 5, (4, 1))
errplot = np.zeros(30)
for i in range(30):
    x1, x2, x3, x4 = forward(n, x, y, w1, w2, w3, w4)
    err = hinge_loss(n, x4, y)
    errplot[i] = err
    print(err)
    w1, w2, w3, w4 = backword(y, x, x1, x2, x3, x4, w1, w2, w3, w4)
plt.plot(np.arange(0, 30), errplot, label="MSE Error")
plt.title('Hinge loss error plot')
plt.show()