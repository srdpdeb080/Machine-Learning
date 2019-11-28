# Import packages

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


from torch import nn
from torch import optim


def calculateLossOverTraining(learningrate, momentum=0.9, epochs=15):
    # Download and load the training data
    trainset = datasets.MNIST('/data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Layer details for the neural network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=momentum)
    time0 = time()
    loss_list = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
            loss_list.append(running_loss/len(trainloader))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)
    return loss_list


def loss_plot(losses, legend = ""):
    max_epochs = len(losses)
    times = list(range(1, max_epochs+1))
    # plt.figure(figsize=(30, 7))
    plt.xlabel("epochs")
    plt.ylabel("cross-entropy loss")
    plt.plot(times, losses, label=legend)
    plt.legend()


learning_rate_list = [10, 1, 1e-1, 1e-2, 1e-4, 1e-5]

lossvalues = {}

for learning_rate in learning_rate_list : 
    print('learning rate is', learning_rate)
    lossvaluelist = calculateLossOverTraining(learning_rate)
    lossvalues[learning_rate] = lossvaluelist
    legend = "Learning Rate is {}".format(learning_rate)
    loss_plot(lossvalues[learning_rate],legend )
    
plt.title('Loss vs Epoch (Momemtum : 0.9)')
plt.show()

#############################################################################################################

#### WEIGHT INITIALIZATION
# Download and load the training data
##########
###### FOR NORMAL W
print('For normal w')
trainset = datasets.MNIST('/data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

    # Build a feed-forward network

w = torch.empty(784, 128)
nn.init.normal_(w)
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
time0 = time()
normal_w_lost_list = []
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # Training pass
        optimizer.zero_grad()
            
        output = model(images)
        loss = criterion(output, labels)
            
        #This is where the model learns by backpropagating
        loss.backward()
            
        #And optimizes its weights here
        optimizer.step()
            
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        normal_w_lost_list.append(running_loss/len(trainloader))
print("\nTraining Time (in minutes) =",(time()-time0)/60)



###### FOR Uniform W
print('FOr Uniform W')
trainset = datasets.MNIST('/data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

    # Build a feed-forward network

w = torch.empty(784, 128)
nn.init.uniform_(w)
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
time0 = time()
uniform_w_lost_list = []
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # Training pass
        optimizer.zero_grad()
            
        output = model(images)
        loss = criterion(output, labels)
            
        #This is where the model learns by backpropagating
        loss.backward()
            
        #And optimizes its weights here
        optimizer.step()
            
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        uniform_w_lost_list.append(running_loss/len(trainloader))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


## FOr xavirers initialization
print('FOr Xaviers Init W')
trainset = datasets.MNIST('/data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

    # Build a feed-forward network

w = torch.empty(784, 128)
nn.init.xavier_normal_(w)
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
time0 = time()
xaviers_w_lost_list = []
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # Training pass
        optimizer.zero_grad()
            
        output = model(images)
        loss = criterion(output, labels)
            
        #This is where the model learns by backpropagating
        loss.backward()
            
        #And optimizes its weights here
        optimizer.step()
            
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        xaviers_w_lost_list.append(running_loss/len(trainloader))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

## PLOTTING
loss_plot(xaviers_w_lost_list,'W initialized xaviers')
loss_plot(normal_w_lost_list,'W initialized normal')
loss_plot(uniform_w_lost_list,'W initialized uniform')
plt.title('Loss vs Epochs')
plt.show()
