import numpy as np


def lr(z):
    a = 1 / (1 + np.exp(-z))
    return a


### Random values Generation
p = np.random.normal(0, 1, 500) / 10000  # y = 1
q = np.random.normal(0, 2, 500) / 10000  # y = -1
sample = [p, q]
sample = sample
sample_min = np.min(sample)
sample = np.array(sample)
w = 2
eta = 0.1
update_sum = 0

### Gradient descent implementation of Linear Regression
for i in range(2):
    for j in range(500):
        if i == 0:
            update_sum += (1 - sample[i, j] * w) * (-sample[i, j])
        if i == 1:
            update_sum += (-1 - sample[i, j] * w) * (-sample[i, j])
w = w - 2 * eta * update_sum

while np.abs(update_sum) > 0.001:
    for i in range(2):
        for j in range(500):
            if i == 0:
                update_sum += (1 - sample[i, j] * w) * (-sample[i, j])
            if i == 1:
                update_sum += (-1 - sample[i, j] * w) * (-sample[i, j])
    w = w - 2 * eta * update_sum

### Minimum by Linear regression
v = 0
for i in range(2):
    for j in range(500):
        if i == 0:
            v += (1 - sample[i, j] * w) ** 2
        if i == 1:
            v += (-1 - sample[i, j] * w) ** 2
print(v, '= Value by Linear Regression')
val = 0

### Minimum Value Desired
z = []
for k in sample:
    for i in range(2):
        for j in range(500):
            if i == 0:
                val += (1 - sample[i, j] * k) ** 2
            if i == 1:
                val += (-1 - sample[i, j] * k) ** 2
    z = np.append(z, val)
val = np.min(z)
print(val, "= Minimum Desired Value")

### Gradient Descent implementation of Logistic Regression
update_sum = 0
for i in range(2):
    for j in range(500):
        if i == 0:
            update_sum += (1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
        if i == 1:
            update_sum += (-1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
w = w - 2 * eta * update_sum

while update_sum > 0.001:
    for i in range(2):
        for j in range(500):
            if i == 0:
                update_sum += (1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
            if i == 1:
                update_sum += (-1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
    w = w - 2 * eta * update_sum

### Minimum Value using Logistic Regression
value = 0
for i in range(2):
    for j in range(500):
        if i == 0:
            value += (1 - lr(sample[i, j] * w)) ** 2
        if i == 1:
            value += (-1 - lr(sample[i, j] * w)) ** 2
print(value, "= Value By Logistic Regression")
print(np.abs(v - val), '= Difference between Desired Min and Linear Regression')
print(np.abs(value - val), '= Difference between Desired Min and Logistic Regression')