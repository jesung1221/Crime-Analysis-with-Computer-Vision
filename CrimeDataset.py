import os
import torch
from torch.utils.data import Dataset
import numpy as np

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
                    self.Label.append([int(x) for x in temp[1].split(',')])
                    self.Data.append(np.zeros((60 * 10 * 18, 2)))
                    ofstfrm = -180
                if templen == 1:
                    flag = True
                    ofstfrm += 180
                    ofstps = 0
                if templen == 3:
                    if flag:
                        comp = int(temp[0])
                        flag = False
                    elif int(temp[0]) <= comp:
                        ofstps += 18
                    self.Data[cnt][int(temp[0])+ofstfrm+ofstps,:] = np.array([float(temp[1]), float(temp[2])])
                line = f.readline()

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.Data[idx])
        label = torch.IntTensor(self.Label[idx])
        return(data, label)