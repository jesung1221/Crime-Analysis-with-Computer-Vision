import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

batch_size = 5
FramePSec = 30
TimeWindow = 2
NumPeople = 10
NumKP = 18
Numepoch = 75
losslst = []
accuracylst = []

class CrimeDataset(Dataset):

    def __init__(self, txtf):
        ## cnt is to count each sample (2second of frames), need to start with -1 as 0 is the first 
        cnt = -1
        
        ## flag to track node for different person
        
        ## ofstps == offset of a person 36
        ## ofstfrm == offset of a frame. Each frame hold 360 space in vector. (each frame 10 peopel. skip 10 people node information)
        
        ## templen =2 --> label of the category
        ## templen =1 --> stars (start of a new frame)
        ## templen =3 --> key points.
        
        ## temp[0] = index, temp[1] = x pos, temp[2] = y pos
        
        flag = False
        self.Label = []
        self.Data = []
        with open(txtf) as f:
            line = f.readline()
            while line:
                temp = line.split()
                templen = len(temp)
                if templen == 2:
                    cnt += 1
                    self.Label.append(np.fromstring(temp[1], float, sep = ','))
                    self.Data.append(np.zeros(FramePSec * TimeWindow * NumPeople * NumKP * 2))
                    ofstfrm = -NumPeople * NumKP * 2
                if templen == 1:
                    flag = True
                    ofstfrm += NumPeople * NumKP * 2
                    ofstps = 0
                if templen == 3:
                    if flag:
                        flag = False
                    elif int(temp[0]) <= comp:
                        ofstps += NumKP * 2
                    self.Data[cnt][int(temp[0])*2+ofstfrm+ofstps] = float(temp[1])
                    self.Data[cnt][int(temp[0])*2+ofstfrm+ofstps+1] = float(temp[2])
                    comp = int(temp[0])
                line = f.readline()

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.Data[idx])
        label = torch.FloatTensor(self.Label[idx])
        return(data, label)

crimeset = CrimeDataset('Output.txt')

## data augmentation
## data[0] is the actual data;  data[0][j] 
## data[1] is the label 

for i, data in enumerate(crimeset, 0):
    temp = np.zeros(FramePSec * TimeWindow * NumPeople * NumKP * 2)
    for j in range(FramePSec * TimeWindow * NumPeople * NumKP * 2):
        if data[0][j] != 0:
            temp[j] = data[0][j]+np.random.uniform(-0.03,0.03)
        if temp[j] > 1:
            temp[j] = 1
        if temp[j] < 0:
            temp[j] = 0
    crimeset.Label.append(data[1])   ## append same label 
    crimeset.Data.append(temp)       ## append data
    if i == 399:
        break

## training dataset 80%, test set 20%
TrainSize = int(len(crimeset) * 0.8)
TestSize = len(crimeset) - TrainSize
TrainSet, TestSet = torch.utils.data.random_split(crimeset, [TrainSize, TestSize])

Trainloader = DataLoader(TrainSet, batch_size = batch_size, shuffle = True)
Testloader = DataLoader(TestSet, batch_size = batch_size, shuffle = False)

classes = ['Abuse', 'Arson', 'Assault', 'Shooting', 'Vandalism', 'Burglary', 'Fighting', 'Robbery', 'Shoplifting', 'Stealing']

class CrimeFC(nn.Module):
    def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(FramePSec * TimeWindow * NumPeople * NumKP * 2, 1020)
            self.fc2 = nn.Linear(1020, 510)
            self.fc3 = nn.Linear(510, 50)
            self.fc4 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CrimeFCN = CrimeFC()
CrimeFCN.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CrimeFCN.parameters(), lr=0.001, momentum=0.9)
for epoch in range(Numepoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(Trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = CrimeFCN(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


## dictionary ["class": counter]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

## model accuracy in each epoch 
    with torch.no_grad():
        for data in Testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = CrimeFCN(inputs)
            _, groundtruths = torch.max(labels,1)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for groundtruth, prediction in zip(groundtruths, predictions):
                if groundtruth == prediction:
                    correct_pred[classes[groundtruth]] += 1
                total_pred[classes[groundtruth]] += 1
    losslst.append(running_loss / 30)
    accuracy = []
    for classname, correct_count in correct_pred.items():
#        accuracy += (100 * float(correct_count) / total_pred[classname])
#    accuracylst.append(accuracy / len(classes))
        accuracy.append(100 * float(correct_count) / total_pred[classname])
    accuracylst.append(accuracy)


plt.cla()
plt.title('Train Loss vs. Epoch', fontsize = 20)
plt.plot(losslst)
plt.xlabel('Num. of Epoch', fontsize = 16)
plt.ylabel('Train Loss', fontsize = 16)
plt.savefig('TrainLoss.png')

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
plt.sca(ax1)
plt.title('Accuracy vs. Epoch', fontsize = 16)
lst = [accu[0] for accu in accuracylst]
plt.plot(lst, label = 'Abuse')
lst = [accu[1] for accu in accuracylst]
plt.plot(lst, label = 'Arson')
lst = [accu[2] for accu in accuracylst]
plt.plot(lst, label = 'Assault')
lst = [accu[3] for accu in accuracylst]
plt.plot(lst, label = 'Shooting')
lst = [accu[4] for accu in accuracylst]
plt.plot(lst, label = 'Vandalism')
lst = [accu[5] for accu in accuracylst]
plt.plot(lst, label = 'Burglary')
lst = [accu[6] for accu in accuracylst]
plt.plot(lst, label = 'Fighting')
lst = [accu[7] for accu in accuracylst]
plt.plot(lst, label = 'Robbery')
lst = [accu[8] for accu in accuracylst]
plt.plot(lst, label = 'Shoplifting')
lst = [accu[9] for accu in accuracylst]
plt.plot(lst, label = 'Stealing')
plt.legend(loc = 'best', fontsize = 6)
#plt.plot(np.linspace(20, 100, 80), accuracylst[20:])
plt.xlabel('Num. of Epoch', fontsize = 10)
plt.ylabel('Accuracy', fontsize = 10)

plt.sca(ax2)
Numsample = [50, 61, 108, 54, 127, 553, 194, 362, 217, 262]
plt.bar(range(len(Numsample)), Numsample, tick_label = classes)
plt.xticks(rotation = 45)
plt.ylabel('Number of Samples', fontsize = 10)
plt.tight_layout()
plt.savefig('Accuracy_NumSamples.png')
torch.save(CrimeFCN.state_dict(),'CrimeClassifer_FC.pth')