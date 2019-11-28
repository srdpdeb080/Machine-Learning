import numpy as np


def dist(x1, x2):
    d = np.sqrt(np.square(x1[0] - x2[0]) + np.square(x1[1] - x2[1]))
    return d


X = np.array([[1, 6],
              [5, 6],
              [8, 6],
              [5, 3],
              [1, 0],
              [5, 0],
              [8, 0]])
init1 = [3, 3]
init2 = [7, 3]
c1 = np.zeros(X.shape)
c2 = np.zeros(X.shape)
p1 = 0
p2 = 0
f = 0
for j in range(5):
    for i in range(X.shape[0]):
        if dist(X[i], init1) > dist(X[i], init2):
            c2[i] = X[i]
        else:
            c1[i] = X[i]
    p1 = [np.mean(c1[:, 0]), np.mean(c1[:, 1])]
    p2 = [np.mean(c2[:, 0]), np.mean(c2[:, 1])]
    print()
    if p1 == init1 and p2 == init2:
        f = 1
        # break
    init1 = p1
    init2 = p2
    print(p1, p2)

if f == 1:
    print('done')