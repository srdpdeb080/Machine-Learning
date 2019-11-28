import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions


def point(c, r):
    i = c[0]
    j = c[1]
    px = []
    py = []
    for d in range(0, 360):
        x = i + r * np.cos(np.deg2rad(d))
        y = j + r * np.sin(np.deg2rad(d))
        px = np.append(px, x)
        py = np.append(py, y)
    return np.array(px), np.array(py)


data = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
data = np.array(data)
z10, z11 = point(data[0], 0.01)
z1 = [z10, z11]
z1 = np.reshape(z1, (2, 360))
z20, z21 = point(data[1], 0.01)
z2 = [z20, z21]
z2 = np.reshape(z2, (2, 360))
z30, z31 = point(data[2], 0.01)
z3 = [z30, z31]
z3 = np.reshape(z3, (2, 360))
z40, z41 = point(data[3], 0.01)
z4 = [z40, z41]
z4 = np.reshape(z4, (2, 360))
z = np.hstack((z1, z2, z3, z4))
y = np.zeros(1440)
y[0:720] = 1
y[721:1440] = -1

c = [0.2, 1, 10]
for k in range(len(c)):
    plt.subplot(1, len(c), k+1)
    plt.title('poly kernel C:' + str(c[k]))
    for i in range(len(y)):
        if y[i] == 1:
            plt.plot(z[0, i], z[1, i], 'r*')
        else:
            plt.plot(z[0, i], z[1, i], 'g*')
    clf = svm.SVC(kernel='poly', C=c[k])
    y = y.astype(np.integer)
    clf.fit(z.T, y)
    plot_decision_regions(z.T, y, clf=clf, legend=0)
plt.show()


for k in range(len(c)):
    plt.subplot(1, len(c), k+1)
    plt.title('rbf kernel C:'+str(c[k]))
    for i in range(len(y)):
        if y[i] == 1:
            plt.plot(z[0, i], z[1, i], 'r*')
        else:
            plt.plot(z[0, i], z[1, i], 'g*')
    clf = svm.SVC(kernel='rbf', C=c[k])
    y = y.astype(np.integer)
    clf.fit(z.T, y)
    plot_decision_regions(z.T, y, clf=clf, legend=0)
plt.show()


for k in range(len(c)):
    plt.subplot(1, len(c), k+1)
    plt.title('linear kernel C:'+str(c[k]))
    for i in range(len(y)):
        if y[i] == 1:
            plt.plot(z[0, i], z[1, i], 'r*')
        else:
            plt.plot(z[0, i], z[1, i], 'g*')
    clf = svm.SVC(kernel='linear', C=c[k])
    y = y.astype(np.integer)
    clf.fit(z.T, y)
    plot_decision_regions(z.T, y, clf=clf, legend=0)
plt.show()