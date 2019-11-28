import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("wine.data")
data = np.array(data)
print(data.shape)
cov = np.cov(data.T)
eval,evct = (np.linalg.eig(cov))
eval = np.array(eval)
print(np.abs(eval))
evct = np.array(evct)
print(evct.shape)
pr_com = evct[: , 0:2]
proj = (data).dot(pr_com)
print(proj)
ev = eval[1:14]
plt.subplot(1,2,1)
# y=np.arange(0,177)
# y=np.array(y)
# y=y.T
# print(y.shape)
plt.plot(np.abs(eval),'rx')
plt.title('Eigen Value spectrum')
plt.subplot(1,2,2)
plt.plot(np.abs(ev),'rx')
plt.title('nearing zero values')
plt.show()
plt.plot(proj[:,0],'rx',label = 'Class 1')
plt.plot(proj[:,1],'bx',label = 'Class 2')
plt.legend(loc = 'upper right')
plt.title('2 D Scatter Plot')
plt.show()
