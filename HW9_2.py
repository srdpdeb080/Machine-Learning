import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(x_val, rate, pr, max_itr):
    itr = 0
    previous = 1
    first = 0
    arr = np.zeros(1000)
    while previous > pr and itr < max_itr:
        x_prev = x_val
        x_val = x_val - rate * 2 * x_prev
        if first < 5:
            print(x_val)
            first = first + 1
        previous = np.abs(x_prev - x_val)
        arr[itr] = previous
        itr = itr + 1
    return itr, arr


# For Calculating Error vs Time
itr, arr = gradientDescent(-2, 0.1, 0.000001, 1000)
plt.plot(np.arange(1, itr + 1), arr[:itr])
plt.title('Error vs Time')
plt.show()

# For calculating convergence criteria
cr = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
it = np.zeros(5)
for i in range(5):
    it[i], arr = gradientDescent(-2, 0.1, cr[i], 1000)
plt.plot(it, cr)
plt.title('Convergence criteria vs iterations')
plt.show()

# For convergence , divergence, and oscillation
# learning rate = 0.1 for convergence
# learning rate = 1 for divergence
# learning rate = 1.3 for oscillation
itr1, arr1 = gradientDescent(-2, 0.1, 0.000001, 500)
itr2, arr2 = gradientDescent(-2, 1, 0.000001, 50)
itr3, arr3 = gradientDescent(-2, 1.3, 0.000001, 50)

plt.plot(np.arange(1, itr1 + 1), arr1[:itr1])
plt.plot(np.arange(1, itr2 + 1), arr2[:itr2])
plt.plot(np.arange(1, itr3 + 1), arr3[:itr3])
plt.show()