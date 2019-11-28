# Code here
from numpy import linalg as LA
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import pdb
import numpy as np

# #parameters
batch_size = 128

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# #Loading the train set file
dataset = datasets.MNIST(root='./data',
                         transform=preprocess,
                         download=True)

loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        xr = self.decoder(h)
        return xr, h


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using CUDA ', use_cuda)


net = AE()
net = net.to(device)

# #Mean square loss function
criterion = nn.MSELoss()

# #Parameters
learning_rate = 1e-2
weight_decay = 1e-5

# #Optimizer and Scheduler
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, threshold=0.001, patience=5, verbose=True)

num_epochs = 5

# #Training
epochLoss = []
for epoch in range(num_epochs):
    total_loss, cntr = 0, 0

    for i, (images, _) in enumerate(loader):

        images = images.view(-1, 28*28)
        images = images.to(device)

        # Initialize gradients to 0
        optimizer.zero_grad()

        # Forward pass (this calls the "forward" function within Net)
        outputs, _ = net(images)

        # Find the loss
        loss = criterion(outputs, images)

        # Find the gradients of all weights using the loss
        loss.backward()

        # Update the weights using the optimizer and scheduler
        optimizer.step()

        total_loss += loss.item()
        cntr += 1

    scheduler.step(total_loss/cntr)
    print('Epoch [%d/%d], Loss: %.4f'
          % (epoch+1, num_epochs, total_loss/cntr))
    epochLoss.append(total_loss/cntr)

# _ = loss_plot(epochLoss)

# net = net.to("cpu")
# torch.save(net.state_dict(),'ae_model.ckpt')

# Load model
# net = AE()
# checkpoint = torch.load('ae_model.ckpt')
# net.load_state_dict(checkpoint)
# net = net.to(device)

# Feature Extraction
ndata = len(dataset)
hSize = 2

test_dataset = datasets.MNIST(root='./data',
                              transform=preprocess,
                              download=True)
test_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=False)

iMat = torch.zeros((ndata, 28*28))
rMat = torch.zeros((ndata, 28*28))
featMat = torch.zeros((ndata, hSize))
labelMat = torch.zeros((ndata))
cntr = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(loader):

        images = images.view(-1, 28*28)
        images = images.to(device)

        rImg, hFeats = net(images)

        iMat[cntr:cntr+batch_size, :] = images
        rMat[cntr:cntr+batch_size, :] = (rImg+0.1307)*0.3081

        featMat[cntr:cntr+batch_size, :] = hFeats
        labelMat[cntr:cntr+batch_size] = labels

        cntr += batch_size

        if cntr >= ndata:
            break

# Reconstruction
reconstruction_loss = 0
for i in range(rMat.shape[0]):
    reconstruction_loss += np.abs((iMat[i, :] - rMat[i, :]))

reconstruction_loss = reconstruction_loss / rMat.shape[0]
print('Before reconc', reconstruction_loss)
reconstruction_loss = LA.norm(reconstruction_loss)
print('recons loss    ', reconstruction_loss)

plt.figure()
plt.axis('off')
plt.imshow(rMat[1, :].view(28, 28), cmap='gray')
plt.show()

plt.figure(figsize = (15,5))
# plt.subplot(131)
plt.scatter(featMat[:,0],featMat[:,1],  c = labelMat)
plt.title('AE Scatter Plot')
plt.show()


# PLOTTING
innernodes = [12, 24, 48, 96, 160, 192, 320]
AdamReconsctructionloss = [20.532963, 20.204,
                           19.912914, 19.770552, 19.702402, 19.775549, 19.73]
SGDWITHOUTMomemtumReconsctructionloss = [
    22.192018, 22.17988, 22.197117, 22.208523, 22.0899, 21.208448, 21.261482]
SGDWITHMomemtumReconsctructionloss = [21.771002, 21.7502, 21.7402, 21.7301, 21.009, 20.8083, 20.606339]
RMSpropReconstructionloss = [21.311478, 20.320105,20.216265,20.430355, 19.9176, 19.821,19.564505 ]

plt.plot(innernodes,AdamReconsctructionloss , color="green", label='Adam Reconstruction Loss')
plt.legend()
plt.xlabel('No of hidden nodes')
plt.ylabel('Reconstruction Error')
plt.show()

plt.plot(innernodes,SGDWITHOUTMomemtumReconsctructionloss , color="orange",  label='SGD without Momemtum Reconstruction Loss')
plt.legend()
plt.xlabel('No of hidden nodes')
plt.ylabel('Reconstruction Error')
plt.show()
plt.plot(innernodes,SGDWITHMomemtumReconsctructionloss  , label='SGD  Momemtum 0.9 Reconstruction Loss')
plt.xlabel('No of hidden nodes')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()
plt.plot(innernodes,RMSpropReconstructionloss ,  label='RMS  Prop Reconstruction Loss')
plt.xlabel('No of hidden nodes')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

# plt.plot(innernodes,AdamReconsctructionloss , 'Adam Reconstruction Loss')