import numpy as np
import matplotlib.pyplot as plt

#random = np.random.normal(1,0.5,1000)
def get_line(m,b,x):
    y = m*x + b
    return y
   
def add_noise(mu,sigma):
    noise = np.random.normal(mu,sigma,1000)
    x_list=[]
    y_list=[]
    for noise,i in zip(noise,range(1000)):
        x_list.append(i)
        y_list.append(get_line(2,1,i)+500*noise)
    return x_list,y_list

def update_weights_MAE(m, b, X, Y, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        # Calculate partial derivatives
        # -x(y - (mx + b)) / |mx + b|
        m_deriv += - X[i] * (Y[i] - (m*X[i] + b)) / abs(Y[i] - (m*X[i] + b))

        # -(y - (mx + b)) / |mx + b|
        b_deriv += -(Y[i] - (m*X[i] + b)) / abs(Y[i] - (m*X[i] + b))

    # We subtract because the derivatives point in direction of steepest ascent
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate

    return m, b

def update_weights_Huber(m, b, X, Y, delta, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        # derivative of quadratic for small values and of linear for large values
        if abs(Y[i] - m*X[i] - b) <= delta:
          m_deriv += -X[i] * (Y[i] - (m*X[i] + b))
          b_deriv += - (Y[i] - (m*X[i] + b))
        else:
          m_deriv += delta * X[i] * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
          b_deriv += delta * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
   
    # We subtract because the derivatives point in direction of steepest ascent
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate

    return m, b

def Converge_MAE(x):
    m,b = 2,1
    m_new,b_new=0,0
    while(1):
        m_new,b_new=update_weights_MAE(m,b,x,y,0.001)
        #print(m_new," ",b_new)
        if(np.abs(np.abs(m)-np.abs(m_new))<0.1):
            break;
        m,b=m_new,b_new
    y1=[]
    for i in range(1000):
        y1.append(get_line(m,b,x[i]))
    return y1  

def Converge_Huber(x):
    m,b = 2,1
    m_new,b_new=0,0
    while(1):
        m_new,b_new=update_weights_Huber(m,b,x,y,1,0.001)
        #print(m_new," ",b_new)
        if(np.abs(np.abs(m)-np.abs(m_new))<0.1):
            break;
        m,b=m_new,b_new
    y1=[]
    for i in range(1000):
        y1.append(get_line(m,b,x[i]))
    return y1

x,y = add_noise(1,1)
ymae = Converge_MAE(x)
yhub = Converge_Huber(x)

plt.plot(x,y,'o',color='y')
plt.title('MAE Loss Function')
plt.plot(x,ymae,color='g')
plt.show()

plt.plot(x,y,'o',color='y')
plt.title('Huber Loss Function')
plt.plot(x,yhub,color='g')
plt.show()