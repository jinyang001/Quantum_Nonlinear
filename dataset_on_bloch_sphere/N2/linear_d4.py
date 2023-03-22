import numpy as np
import torch
#from torch import nn
import torchcomplex.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchquantum as tq
import matplotlib
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pi = np.pi
block_points_num = 400
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
Y = np.concatenate((np.ones(block_points_num), np.zeros(block_points_num), np.ones(block_points_num),
                    np.zeros(block_points_num), np.zeros(block_points_num), np.ones(block_points_num),
                    np.zeros(block_points_num), np.ones(block_points_num)), axis=0)

class TorchDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor(X, dtype=torch.float)
        self.label = torch.tensor(Y, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


data = TorchDataset()

train_size, test_size, test_size2 = int(len(data) * 0.8), int(len(data) * 0.1), len(data) - int(len(data) * 0.8)- int(len(data) * 0.1)
train_dataset, test_dataset,test_dataset2 = random_split(data, [train_size, test_size,test_size2])

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader2 = DataLoader(dataset=test_dataset2, batch_size=len(test_dataset2), shuffle=False, num_workers=0, drop_last=False)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['1x1_u3'])
        self.layer1 = nn.Linear(2, 2)
        #self.layer2 = nn.Linear(1, 2)
        # self.backbone = nn.Sequential(
        #     nn.Linear(2, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 2),
        # )

    def forward(self, x):
        # print(x)
        self.encoder(self.q_device, x)
        #print(self.q_device.get_states_1d())
        #temp=self.q_device.get_states_1d()[0][0]
        #print(type(temp))
        #print(self.q_device.get_states_1d().real)

        x = self.layer1(self.q_device.get_states_1d())
        #x = torch.pow(x, 2)
        #x = self.layer2(x)
        #x=self.layer1(x)
        m = nn.Softmax(dim=1)
        x = m(x)
        return x


def train(dataloader, model, criterion, optimizer, device):
    all_losses = []
    correct = 0
    for i, (X, y) in enumerate(dataloader):
        inputs, targets = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(targets)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        all_losses.append(loss)
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    # print(loss)
    train_acc = 100. * correct / len(dataloader.dataset)
    train_loss = float(sum(all_losses)) / len(dataloader)
    return train_loss, train_acc


def test(dataloader, model, criterion, device):
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    # test_loss, correct = 0, 0
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            inputs, targets = X.to(device), y.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets.long()).item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            # correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

        # test_loss /= num_batches
        # correct /= size
        # loss_list.append(test_loss)
        # accuracy_list.append(correct)
        test_acc = 100. * correct / len(dataloader.dataset)
        test_loss /= len(dataloader)
        return test_loss, test_acc
        # print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# def test2(dataloader, model, criterion, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     xoutput=[]
#     youtput=[]
#     with torch.no_grad():
#         for _, (X, y) in enumerate(dataloader):
#             inputs, targets = X.to(device), y.to(device)
#             xoutput.extend(inputs.tolist())
#
#
#             outputs = model(inputs)
#
#             test_loss += criterion(outputs, targets.long()).item()
#             pred = outputs.data.max(1, keepdim=True)[1]
#             youtput.extend(pred.tolist())
#
#
#             correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
#     #print(xoutput)
#     #print(youtput)
#     xoutput=torch.tensor(xoutput)
#     youtput=sum(youtput,[])
#     print(len(xoutput))
#     print(len(youtput))
#     print(youtput)
#     x = xoutput.T[0]
#     x2 = xoutput.T[1]
#     label = youtput
#     colors = ['orange', 'blue']
#     print(x)
#     print(label)
#
#     fig = plt.figure(figsize=(40, 40))
#     # plt.axhline(y=0)
#     plt.scatter(x, x2,s=2, c=label, cmap=matplotlib.colors.ListedColormap(colors),alpha=0.7)
#     plt.show()
# loss_list=[]
# accuracy_list=[]
def test2(dataloader, model, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    xoutput = []
    youtput = []
    toutput = []
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            inputs, targets = X.to(device), y.to(device)
            # inputs = torch.cat((inputs, inputs), 1)
            xoutput.extend(inputs.tolist())
            outputs = model(inputs)
            test_loss += criterion(outputs, targets.long()).item()
            pred = outputs.data.max(1, keepdim=True)[1]
            youtput.extend(pred.tolist())
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            toutput.extend(targets.data.view_as(pred).cpu().tolist())
            #print(targets.data.view_as(pred).tolist())

        test_acc = 100. * correct / len(dataloader.dataset)
        test_loss /= len(dataloader.dataset)
        return xoutput, youtput, toutput, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = NeuralNetwork().to(device)
    criterion = F.nll_loss
    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    print(model)
    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)
    epochs = n_epochs
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        tr_loss, tr_acc = train(train_loader, model, criterion, optimizer, device)
        print(optimizer.param_groups[0]['lr'])

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        te_loss, te_acc = test(test_loader, model, criterion, device)
        test_loss.append(te_loss)
        test_acc.append(te_acc)
        scheduler.step()
    print("Done!")
    xoutput, youtput, toutput, test_acc_final = test2(test_loader2, model, criterion, device)
    ####plot
    xoutput = torch.tensor(xoutput)
    youtput = sum(youtput, [])
    toutput = sum(toutput, [])
    toutput = list(map(int, toutput))
    x2 = (xoutput.T[0] / 2).tolist()
    x = xoutput.T[0].tolist()
    # print(x)
    # print(x2)
    # print(np.cos(x2))
    # print(np.sin(x2))
    print("accuarcy")
    print(test_acc_final)
    # sss
    x = np.cos(x2)
    x2 = np.sin(x2)

    plt.figure(figsize=(10, 6))
    for x, x2, youtput, toutput in zip(x, x2, youtput, toutput):
        if youtput == 1:
            if youtput != toutput:
                plt.plot(x, x2, "b^", markersize=4)
            else:
                plt.plot(x, x2, "bo", markersize=2)
        else:
            if youtput != toutput:
                plt.plot(x, x2, "r^", markersize=4)
            else:
                plt.plot(x, x2, "ro", markersize=2)

    plt.axhline(y=0, linestyle="--", color="black", lw=1)
    plt.axvline(x=np.cos(np.pi / 4), linestyle="--", color="black", lw=1)

    plt.title('graph')

    # plt.figure(figsize=(5, 5))
    # # plt.axhline(y=0)
    # plt.scatter(correctx, correctx2, s=2, c=label, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.7)
    # plt.scatter(incorrectx, incorrectx2, s=2 ,c='black', alpha=0.7)
    # plt.title('graph')
    ##
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, epochs + 1), train_acc)
    plt.plot(range(1, epochs + 1), test_acc)
    plt.title('Accuracy in {} epochs'.format(epochs))

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, epochs + 1), train_loss)
    plt.plot(range(1, epochs + 1), test_loss)
    plt.title('Loss in {} epochs'.format(epochs))
    plt.show()


if __name__ == '__main__':
    main()