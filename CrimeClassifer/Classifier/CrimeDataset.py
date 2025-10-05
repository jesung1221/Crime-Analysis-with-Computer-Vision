import numpy as np
import torch
from torch.utils.data import Dataset

FramePSec = 30
TimeWindow = 2
NumPeople = 10
NumKP = 18

class CrimeDataset(Dataset):

    def __init__(self, txtf):
        cnt = -1
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