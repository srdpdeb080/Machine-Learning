import numpy as np
import matplotlib.pyplot as plt


def decision(x, mu, sigma):
    x = np.array(x)
    mu = np.array(mu)
    sigma = np.array(sigma)
    d = (x-mu)
    k = np.linalg.pinv(sigma)
    z = np.dot(d, k)
    z = np.dot(z, d.T)
    z = z*(-1)/2
    p = (1 / (np.power((2 * np.pi), (x.shape[0] / 2)) * (np.power(np.linalg.det(sigma), 0.5)))) * (np.exp(z))
    return p


mu1 = [3, 3]
mu2 = [7, 7]
sigma1 = [[3, 1], [2, 3]]
sigma2 = [[7, 2], [1, 7]]
x1, y1 = np.random.multivariate_normal(mu1, sigma1, 1000).T * 10 / 15
x2, y2 = np.random.multivariate_normal(mu2, sigma2, 1000).T * 10 / 15
sigma11 = np.identity(2) * 3
x11, y11 = np.random.multivariate_normal(mu1, sigma11, 1000).T * 10 / 15
x22, y22 = np.random.multivariate_normal(mu2, sigma11, 1000).T * 10 / 15
print(np.max(y2))
plt.scatter(x1, y1, color='red')
plt.scatter(x2, y2, color='green')
plt.title('For second case')
plt.show()
plt.scatter(x11, y11, color='red')
plt.scatter(x22, y22, color='green')
plt.title('For second case')
plt.show()

d1 = decision([1, 2], mu1, sigma1)
d2 = decision([1, 2], mu2, sigma2)
# print(d1)
# print(d2)

if d1 > d2:
    print('Belongs to first class')
else:
    print('belongs to second class')
