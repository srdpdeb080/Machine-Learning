import numpy as np
import matplotlib.pyplot as plt


def createSet(s, k):
    data = np.random.random_integers(0, 1, (s * k))
    var = 0
    mean = np.zeros(s)
    like = np.zeros(s)
    for i in range(s):
        mean[i] = np.mean(data[i*k : i*k + s-1])
    for i in range(s):
        like[i] = (1/np.sqrt(2*np.pi))*np.exp((-1/2)*(np.square(mean[i])))
    var = np.var(like)
    return var


v = createSet(10, 10)
kdiff = [10, 15, 20, 25, 30, 35]
vK = np.zeros(6)
for i in range(6):
    vK[i] = createSet(10, kdiff[i])
plt.plot(np.arange(1,7), vK)
plt.title('Change With K (S constant)')
plt.show()

sdiff = [10, 15, 20, 25, 30, 35]
vs = np.zeros(6)
for i in range(6):
    vs[i] = createSet(sdiff[i], 10)
plt.plot(np.arange(1,7), vs)
plt.title('Change With S (K constant)')
plt.show()