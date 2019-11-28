import numpy as np
import matplotlib.pyplot as plt


def generateSample(p):
    x = np.random.random_integers(0, 1000, p)
    x = np.sin(x)
    mu = 0
    sigma = 0.5
    c = np.random.normal(mu, sigma, p)
    x = x + c
    return x


def KFoldVal(y, n):
    mean = np.mean(y)
    var = np.var(y)
    l = y.shape[0]
    # print(l)
    meanc = []
    varc = []
    for k in range(1, n + 1):
        p = int(l / k)
        q = l // p
        # print(q)
        x = 0
        m = 0
        v = 0
        matC = []
        matV = []
        for i in range(p):
            c = y[x: q + x]
            # print(x, x+q)
            x = x + q
            m = np.mean(c)
            v = np.var(c)
            matC.append(m)
            matV.append(v)
        m = np.mean(matC) - mean
        # print(m)
        v = np.var(matV) - var
        # print(v)
        meanc.append(m)
        varc.append(v)
        # meanc = meanc - mean
    x = list(range(1, n + 1))
    plt.scatter(x, meanc)
    plt.ylabel('K')
    plt.ylabel('mean')
    plt.title('Plot for mean')
    plt.show()

    plt.scatter(x, varc)
    plt.ylabel('K')
    plt.ylabel('Variance')
    plt.title('Plot for Variance')
    plt.show()


y = generateSample(100)
KFoldVal(y, 50)
# print(y.shape)

y = generateSample(10000)
KFoldVal(y, 50)
