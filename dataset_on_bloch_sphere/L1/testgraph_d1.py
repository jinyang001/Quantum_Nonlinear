import qutip
import numpy as np
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchquantum as tq
import matplotlib.pyplot as plt
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pi = np.pi
block_points_num = 4
lam = np.zeros(block_points_num)

theta1 = np.random.uniform(0, pi / 2, block_points_num)
phi1 = np.random.uniform(pi, 3 * pi / 2, block_points_num)
block1 = np.array([theta1, phi1, lam])

theta2 = np.random.uniform(0, pi / 2, block_points_num)
phi2 = np.random.uniform(pi / 2, pi, block_points_num)
block2 = np.array([theta2, phi2, lam])

theta3 = np.random.uniform(0, pi / 2, block_points_num)
phi3 = np.random.uniform(3 * pi / 2, 2 * pi, block_points_num)
block3 = np.array([theta3, phi3, lam])

theta4 = np.random.uniform(0, pi / 2, block_points_num)
phi4 = np.random.uniform(0, pi / 2, block_points_num)
block4 = np.array([theta4, phi4, lam])

theta5 = np.random.uniform(pi / 2, pi, block_points_num)
phi5 = np.random.uniform(pi, 3 * pi / 2, block_points_num)
block5 = np.array([theta5, phi5, lam])

theta6 = np.random.uniform(pi / 2, pi, block_points_num)
phi6 = np.random.uniform(pi / 2, pi, block_points_num)
block6 = np.array([theta6, phi6, lam])

theta7 = np.random.uniform(pi / 2, pi, block_points_num)
phi7 = np.random.uniform(3 * pi / 2, 2 * pi, block_points_num)
block7 = np.array([theta7, phi7, lam])

theta8 = np.random.uniform(pi / 2, pi, block_points_num)
phi8 = np.random.uniform(0, pi / 2, block_points_num)
block8 = np.array([theta8, phi8, lam])

X = np.concatenate((block1.T, block2.T, block3.T, block4.T, block5.T, block6.T, block7.T, block8.T), axis=0)
Y = np.concatenate((np.ones(block_points_num), np.ones(block_points_num), np.ones(block_points_num),
                    np.ones(block_points_num), np.zeros(block_points_num), np.zeros(block_points_num),
                    np.zeros(block_points_num), np.zeros(block_points_num)), axis=0)


class TorchDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor(X, dtype=torch.float)
        self.label = torch.tensor(Y, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


data = TorchDataset()

train_size, test_size = int(len(data) * 1), len(data) - int(len(data) * 1)
train_dataset, test_dataset = random_split(data, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

theta111=[]
phi111=[]
lam111=[]
theta000=[]
phi000=[]
lam000=[]
for batch, (X, y) in enumerate(train_loader):
    print("batch:", batch)
    print(f'X.shape:{X.shape}, y.shape:{y.shape}')
    Data, Label = X, y
    print("data:", Data)
    print("Label:", Label)
    for i, label in enumerate(Label):
        if label:
            theta111.append(Data[i][0])
            phi111.append(Data[i][1])
            lam111.append(Data[i][2])
        else:
            theta000.append(Data[i][0])
            phi000.append(Data[i][1])
            lam000.append(Data[i][2])
    break
print(theta111)
print(phi111)
#theta111=torch.tensor(theta111)
print(theta000)
print(phi000)

x1 = np.cos(phi111)*np.sin(theta111)
y1 = np.sin(phi111)*np.sin(theta111)
z1 = np.cos(theta111)
print(x1)
x0 = np.cos(phi000)*np.sin(theta000)
y0 = np.sin(phi000)*np.sin(theta000)
z0 = np.cos(theta000)

xtest = np.cos(0)*np.sin(np.pi/2)
ytest = np.sin(0)*np.sin(np.pi/2)
ztest = np.cos(np.pi/2)

pi=np.pi
b = qutip.Bloch()
b3d = qutip.Bloch3d()
b.make_sphere()
b.render()
#b.show()
pnt = [x0, y0, z0]
pnt1 = [x1, y1, z1]
#b.add_points(pnt)
b.add_points(pnt1)
b.render()
#b.show()


b.clear()
temp1 = np.array(np.random.uniform(0, pi / 2, 30)).T
temp2 = np.array(np.random.uniform(pi / 2, pi, 30)).T
theta = np.concatenate((temp1,temp2), axis=0)
print(theta)

phi = np.concatenate((np.array(np.random.uniform(pi, pi, 30)).T,np.array(np.random.uniform(pi,pi, 30)).T), axis=0)
theta2 = np.concatenate((np.array(np.random.uniform(0, pi / 2, 30)).T,np.array(np.random.uniform(pi / 2, pi, 30)).T), axis=0)
phi2 = np.concatenate((np.array(np.random.uniform(0, 0, 30)).T,np.array(np.random.uniform(0,0, 30)).T), axis=0)

x3 = np.random.uniform(pi / 2, 0, 20)
th = np.linspace(0, 2*pi, 20)
temp=np.pi
xp = np.cos(phi)*np.sin(theta)
yp = np.sin(phi)*np.sin(theta)
zp = np.cos(theta)

xp2 = np.cos(phi2)*np.sin(theta2)
yp2 = np.sin(phi2)*np.sin(theta2)
zp2 = np.cos(theta2)

x = np.cos(0)*np.sin(np.pi/2)
y = np.sin(0)*np.sin(np.pi/2)
z = np.cos(np.pi/2)

pnt = [x, y, z]
pnts = [xp, yp, zp]
pnts2 = [xp2, yp2, zp2]

b.add_points(pnts)
b.add_points(pnts2)
# b.add_points(pnt)
#b.add_points(pnts2)
up = qutip.basis(2, 0)
print(up)
b.render()
b.show()