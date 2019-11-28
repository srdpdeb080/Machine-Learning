from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np


def computeErrorForaSample(w, k, y, d=5):
    error = 0
    for i in range(1, d+1):
        error += w[i]*y[k-i]
    error = y[k]-error
    return error


def computeDerivativeComponent(w, y, d=5):
    derivative = 0
    for i in range(d, len(y)):
        ytemp = []
        for k in range(1, d+1):
            ytemp.append(y[i-k])
        ytemp = np.array(ytemp)
        derivative += computeErrorForaSample(w, i, y,d) * ytemp
    return derivative


def SLP(y, d=5, maxtrials=5000, learning_Rate=0.00001):
    w = [0]
    for i in range(1,d+1):
        w.append(1)
    trials = 0
    derivative = computeDerivativeComponent(w, y,d)
    # while((trials < maxtrials)):
    while((abs(derivative[0]) > 0.01) and trials < maxtrials):
        w[1:] = w[1:] + learning_Rate * derivative
        trials += 1
        derivative = computeDerivativeComponent(w, y,d)
    print('W is ', w)
    print('Derivative   ', derivative)
    print('trials', trials)
    return w


def generateData(samplepoints, alphas, startdata, d=5, mu=0, sigma=1):
    # datapoints = startdata.copy()
    datapoints = []
    for item in startdata:
        datapoints.append(item)
    randomnoise = np.random.normal(mu, sigma, samplepoints)
    for i in range(len(datapoints), samplepoints):
        nextdatapoint = randomnoise[i]
        for j in range(1, d+1):
            nextdatapoint += alphas[j]*datapoints[i-j]
        datapoints.append(nextdatapoint)
    return datapoints[d:]


alphas = np.array([0, 0.1, 0.4, 0.3, 0.5, 0.2])
samplepoints = 25
d = 5
data = [1, 2, 3, 4, 5]
y = generateData(samplepoints, alphas, data)
data = np.append(data, y)
prediectedw = SLP(data)

moresamplepoints = 40
truedata = generateData(moresamplepoints, alphas, data[len(data)-d:len(data)])

predictedY = generateData(moresamplepoints, prediectedw,
                          data[len(data)-d:len(data)])
startPredicted = len(data)
endPredicted = startPredicted + len(predictedY)
plt.scatter(range(len(data)), data, c="green",
            edgecolor='black', linewidth='1', s=35, zorder=2)
plt.plot(range(len(data)), data, zorder=1, c="green", label='Training Data')
plt.scatter(range(startPredicted, endPredicted), truedata,
            c="orange", edgecolor='black', linewidth='1', s=35, zorder=2)
plt.plot(range(startPredicted, endPredicted), truedata,
         zorder=1, c="orange", label='True Values')
plt.scatter(range(startPredicted, endPredicted), predictedY,
            c="blue", edgecolor='black', linewidth='1', s=35, zorder=2)
plt.plot(range(startPredicted, endPredicted), predictedY,
         zorder=1, c="blue", label='Predicted')
plt.title("alpha1 : {} alpha2 : {} alpha3 : {} alpha4 : {} alpha5 : {} ".format(
    alphas[1], alphas[2], alphas[3], alphas[4], alphas[5]))
plt.xlabel('k (instance) ')
plt.ylabel('x(k)')
plt.legend()
plt.show()


def findLossVaryingD(y, drange=range(5, 6)):
    errorlist = []
    for d in drange:
        predictedw = SLP(y,d)
        error = 0
        for k in range(d, len(y)):
            currentsampleerror = computeErrorForaSample(predictedw, k, y, d)
            error += currentsampleerror**2
        print('W', predictedw)
        print('Error : ', error, 'd : ', d)
        errorlist.append(error)
    return errorlist


drange = range(2,20)
errorList = findLossVaryingD(data,drange)
plt.scatter(drange, errorList, c="red",
            edgecolor='black', linewidth='1', s=35, zorder=2)
plt.plot(drange, errorList, zorder=1, c="red", label='Loss')
plt.grid(color='blue', linestyle='--', linewidth=1, alpha=0.1)
plt.xlabel('d')
plt.ylabel('Loss')
plt.title('Loss vs d')
plt.legend()
plt.show()