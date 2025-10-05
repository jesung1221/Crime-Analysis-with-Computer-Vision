import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

FramePSec = 30
TimeWindow = 2
NumPeople = 10
NumKP = 18
ofstfrm = 21240

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
CrimeFCN.load_state_dict(torch.load('../Classifer/CrimeClassifer_FC.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
CrimeFCN.to(device)

classes = ['Abuse', 'Arson', 'Assault', 'Shooting', 'Vandalism', 'Burglary', 'Fighting', 'Robbery', 'Shoplifting', 'Stealing']


## 21600 is the input vectore size. size of the sample. 
with open('../output/Output_trial.txt') as f:
    predictions = []
    vsample = np.zeros(FramePSec * TimeWindow * NumPeople * NumKP * 2)
    with torch.no_grad():
        line = f.readline()
        while line:
            temp = line.split()
            
            
            if len(temp) == 1:
                flag = True
                output = CrimeFCN(torch.Tensor(vsample).to(device))
                _,prediction = torch.max(output, 0)
                predictions.append(classes[prediction])
                ofstps = 0
                
                ## circular shift and empty;; number of keypoints * number of people * 2 = 360
                ## FIFO Sliding window implementation
                vsample = np.roll(vsample, -360)
                vsample[21240:21600] = 0
            if len(temp) == 3:
                if flag:
                    flag = False
                elif int(temp[0]) <= comp:
                    ofstps += NumKP * 2
                vsample[int(temp[0])*2+ofstfrm+ofstps] = float(temp[1])
                vsample[int(temp[0])*2+ofstfrm+ofstps+1] = float(temp[2])
                comp = int(temp[0])
            line = f.readline()

output_path = ''

## Vtarget represents the video. var of video class which holds the properties of the video. 

Vtarget = cv2.VideoCapture(output_path)
fps = int(round(Vtarget.get(5)))
width = int(Vtarget.get(3))
height = int(Vtarget.get(4))
totalf = int(Vtarget.get(7))
Vwrite = cv2.VideoWriter('../output/Classified.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(width,height))

##writing the label on the corner of the video
## ret = 1 video didn't end. ret = 0 video ended. 
cnt = 0
while True:
    ret, imgtarget = Vtarget.read()         ## read the video frame by frame
    if ret:
        ## drop the first 30 frames
        if cnt <= (totalf - 30):
            txtsize = cv2.getTextSize(predictions[cnt + 29], cv2.FONT_HERSHEY_COMPLEX, 2, 2)
            imgwrite = cv2.putText(imgtarget, predictions[cnt + 29], (width-txtsize[0][0],txtsize[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
        else:
            imgwrite = imgtarget
        cnt += 1
        Vwrite.write(imgwrite)
    else:
        break

Vtarget.release()
Vwrite.release()