import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CrimeFC(nn.Module):
    def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 50)
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
