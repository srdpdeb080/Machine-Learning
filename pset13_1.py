import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_logistic(x, y, w):
    m = x.shape[0]
    z = np.dot(w, x.T)
    
    h = sigmoid(z)
    h = h.reshape(len(h), 1)
    p = (-1 / m) * (np.dot(x.T, (h - y)))
    return p.reshape(1, len(p))


def gradient_descent_logistic(x, y, w):
    n = 0.3
    wOld = w
    wNew = 0
    itr = 7000
    c = 0
    while c < itr:
        wNew = wOld - n * gradient_logistic(x, y, wOld)[0]
        print(wNew)
        c += 1
        if np.array_equal(wNew, wOld):
            break
        wOld = wNew
    print(c)
    return wNew


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
    itr = 7000
    for i in range(itr):
        sum = np.zeros(x.shape[1])
        for j in range(x.shape[0]):
            if y[j] * val(wOld, x[j]) < 0:
                sum += x[j] * y[j]
        wNew = wOld + n * sum
        # Convergence Criteria is satisfied or no of iteration < itr
        if np.array_equal(wOld, wNew):
            break
        print(wNew, wOld, sum)
        wOld = wNew
    return wNew


x = np.array([[1, 1, 1],[2, 1, 1],[3, 4, 1],[4, 4, 1]])
y = np.array([[1],[1],[-1],[-1]])
wOld = np.random.random_integers(1, 10, 3)
w = gradient_descent_logistic(x, y, wOld)
print('Final Value of W:', w)
plt.plot(x, w[0] * x / w[1] + w[2] / w[1], linestyle='solid')
plt.scatter([1, 2], [1, 1])
plt.scatter([3, 4], [4, 4])
plt.title('Logistic Regression')
plt.show()

w = Perceptron(x, y, wOld)
print('Final Value of w:', w)
plt.plot(x, w[0] * x / w[1] + w[2] / w[1], linestyle='solid')
plt.scatter([1, 2], [1, 1])
plt.scatter([3, 4], [4, 4])
plt.title('Perceptron')
plt.show()