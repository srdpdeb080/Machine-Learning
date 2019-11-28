from sympy import symbols
import sympy
from sympy.plotting import plot_implicit
import numpy as np
import matplotlib.pyplot as plt

w1 = np.array([[0, 0, 2 , 3, 3, 2, 2],
      [1, 1, 0, 2, 3, 2, 0]])
w2 = np.array([[7, 8, 9, 8, 7, 8, 7],
      [7, 6, 7, 10, 10, 9, 11]])

mean1 = [np.mean(w1[0]), np.mean(w1[1])]
mean2 = [np.mean(w2[0]), np.mean(w2[1])]
cov1 = np.cov(w1)
cov2 = np.cov(w2)
print(mean1, mean2)
print(cov1)
print(cov2)
w1 = np.array(w1.T - mean1)
print(w1)
x1 = symbols('x1')
x2 = symbols('x2')
p1 = (-1/2)*np.dot(np.dot([x1, x2], np.linalg.inv(cov1)), [[x1], [x2]]) - (1/2)*np.log(np.linalg.det(cov1))
p2 = (-1/2)*np.dot(np.dot([x1, x2], np.linalg.inv(cov2)), [[x1], [x2]]) - (1/2)*np.log(np.linalg.det(cov2))
p3 = p1 - p2
# print(p1)
# print(p2)
print(p3)
f = sympy.solve(p3, [x2])
print(f)
x = (np.arange(150.)-5)/10
y = sympy.lambdify(x1, f, 'numpy')(x)
print(x)
plt.scatter(w1[0], w1[1])
plt.scatter(w2[0], w2[1])
plt.plot(x, np.transpose(y[0]))
plt.show()
