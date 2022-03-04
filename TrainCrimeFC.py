import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import CrimeDataset

batch_size = 4

crimeset = CrimeDataset("Output.txt")

trainloader = DataLoader(crimeset, batch_size=batch_size, shuffle=True)

class CrimeFC(nn.Module):
    def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(60 * 10 * 18, 1020)
            self.fc2 = nn.Linear(1020, 100)
            self.fc3 = nn.Linear(100, 50)
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

CrimeFCN = CrimeFC()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CrimeFCN.parameters(), lr=0.001, momentum=0.9)

