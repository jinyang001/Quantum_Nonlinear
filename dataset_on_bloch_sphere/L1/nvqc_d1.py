import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import torchquantum as tq
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchquantum.plugins import (tq2qiskit_expand_params,
                                  tq2qiskit,
                                  tq2qiskit_measurement,
                                  qiskit_assemble_circs)

import torchquantum.functional as tqf
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pi = np.pi
x1 = np.random.uniform(-pi / 2, 0, 800)

upper_right = np.array([x1])

x2 = np.random.uniform(0, pi / 2, 800)

upper_left = np.array([x2])

x3 = np.random.uniform(pi / 2, pi, 800)

down_left = np.array([x3])

x4 = np.random.uniform(-pi, -pi / 2, 800)

down_right = np.array([x4])

X = np.concatenate((upper_right.T, upper_left.T, down_left.T, down_right.T), axis=0)
Y = np.concatenate((np.ones(800), np.zeros(800), np.zeros(800), np.ones(800)), axis=0)

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

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1
            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)
            self.crx1 = tq.CRX(has_params=True, trainable=True)

            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.rz2 = tq.RZ(has_params=True, trainable=True)
            self.crx2 = tq.CRX(has_params=True, trainable=True)

            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.ry3 = tq.RY(has_params=True, trainable=True)
            self.crx3 = tq.CRX(has_params=True, trainable=True)

            self.crx4 = tq.CRX(has_params=True, trainable=True)
            self.crx5 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=0)
            #self.rz0(self.q_device, wires=0)
            # self.crx0(self.q_device, wires=[0, 1])
            self.rx1(self.q_device, wires=0)
            self.ry1(self.q_device, wires=0)
            #self.rz1(self.q_device, wires=0)
            # self.crx1(self.q_device, wires=[2, 3])
            #
            #self.rx2(self.q_device, wires=0)
            # self.ry2(self.q_device, wires=0)
            self.rz2(self.q_device, wires=0)
    class QLayer2(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1
            # gates with trainable parameters
            self.l2rx0 = tq.RX(has_params=True, trainable=True)
            self.l2ry0 = tq.RY(has_params=True, trainable=True)
            #self.rz0 = tq.RZ(has_params=True, trainable=True)
           # self.crx0 = tq.CRX(has_params=True, trainable=True)

            self.l2rx1 = tq.RX(has_params=True, trainable=True)
            self.l2ry1 = tq.RY(has_params=True, trainable=True)
           # self.rz1 = tq.RZ(has_params=True, trainable=True)
           # self.crx1 = tq.CRX(has_params=True, trainable=True)

            # self.rx2 = tq.RX(has_params=True, trainable=True)
            # self.ry2 = tq.RY(has_params=True, trainable=True)
            # self.rz2 = tq.RZ(has_params=True, trainable=True)
            # self.crx2 = tq.CRX(has_params=True, trainable=True)
            #
            # self.rx3 = tq.RX(has_params=True, trainable=True)
            # self.ry3 = tq.RY(has_params=True, trainable=True)
            # self.crx3 = tq.CRX(has_params=True, trainable=True)
            #
            # self.crx4 = tq.CRX(has_params=True, trainable=True)
            # self.crx5 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.l2rx0(self.q_device, wires=0)
            self.l2ry0(self.q_device, wires=0)
            #self.rz0(self.q_device, wires=0)
            # self.crx0(self.q_device, wires=[0, 1])
            self.l2rx1(self.q_device, wires=0)
            self.l2ry1(self.q_device, wires=0)
            #self.rz1(self.q_device, wires=0)
            # self.crx1(self.q_device, wires=[2, 3])
            #
            # self.rx2(self.q_device, wires=0)
            # self.ry2(self.q_device, wires=0)
            #self.rz2(self.q_device, wires=0)

    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['1x1_ry'])
        self.encoder2 = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['1x1_ry'])
        self.q_layer = self.QLayer()
        self.q_layer2 = self.QLayer2()
        self.measure = tq.MeasureAll(tq.PauliZ)

    # def forward(self, x):
    #     print(x[0])
    #     self.encoder(self.q_device, x)
    #     print(self.q_device.get_states_1d()[0])
    #     self.q_layer(self.q_device)
    #     encoder_circs = tq2qiskit_expand_params(self.q_device, x,
    #                                             self.encoder.func_list)
    #     q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
    #     measurement_circ = tq2qiskit_measurement(self.q_device,
    #                                              self.measure)
    #     assembled_circs = qiskit_assemble_circs(encoder_circs,
    #                                             q_layer_circ,
    #                                             measurement_circ)
    #     #print(measurement_circ)
    #     print(assembled_circs[0].draw())
    #     assembled_circs[0].draw(output='mpl')
    #     plt.show()
    #
    #     ssss
    #     x = self.measure(self.q_device)
    #
    #     x = F.log_softmax(x, dim=1)
    #
    #     return x

    def forward(self, x):

        self.encoder(self.q_device, x)

        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        self.encoder2(self.q_device, x)
        self.q_layer2(self.q_device)
        x = self.measure(self.q_device)

        x = torch.sigmoid(x)

        return x


def train(dataloader, model, criterion, optimizer, device):
    all_losses = []
    correct = 0
    for _, (X, y) in enumerate(dataloader):
        inputs, targets = X.to(device), y.to(device)

        #inputs = torch.cat((inputs, inputs), 1)

        outputs = model(inputs)
        # print('inputs')
        # print(inputs)
        # print('outputs')
        # print(outputs)

        #print(outputs)
        pred=torch.where(outputs>=0.5, 1, 0)

        #print(pred.data.view_as(pred).cpu())
        #print(targets)
        loss = criterion(outputs, targets.reshape(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_losses.append(loss)
        #pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
        #print(correct)


    train_acc = 100. * correct / len(dataloader.dataset)
    train_loss = float(sum(all_losses)) / len(dataloader.dataset)
    return train_loss, train_acc

    # print(loss.item())


def test(dataloader, model, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            inputs, targets = X.to(device), y.to(device)
            #inputs = torch.cat((inputs, inputs), 1)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets.reshape(-1, 1)).item()
            #pred = outputs.data.max(1, keepdim=True)[1]
            pred = torch.where(outputs >= 0.5, 1, 0)
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        test_acc = 100. * correct / len(dataloader.dataset)
        test_loss /= len(dataloader.dataset)
        return test_loss, test_acc


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
            #inputs = torch.cat((inputs, inputs), 1)
            xoutput.extend(inputs.tolist())
            outputs = model(inputs)
            test_loss += criterion(outputs, targets.reshape(-1, 1)).item()
            pred = torch.where(outputs >= 0.5, 1, 0)
            #pred = outputs.data.max(1, keepdim=True)[1]
            youtput.extend(pred.tolist())
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            toutput.extend(targets.data.view_as(pred).cpu().tolist())
        q_layer_circ = tq2qiskit(model.q_device, model.q_layer)
        # print(inputs)
        # print(model.q_device.get_states_1d())
        # print(q_layer_circ)
        # print(model.q_layer.get_unitary(model.q_device))
        # print('x')
        # print(xoutput)
        # sss

        test_acc = 100. * correct / len(dataloader.dataset)
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
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    criterion = nn.BCELoss()
    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    epochs = n_epochs
    print(device)
    q_layer_circ = tq2qiskit(QFCModel().q_device, QFCModel().q_layer)

    q_layer_circ.draw(output='mpl')
    plt.show()
    # sss
    print('start')
    for epoch in range(epochs):
        # train
        print(f"Epoch {epoch}:")
        # tr_loss, tr_acc = train(dataflow, model, device, optimizer)
        tr_loss, tr_acc = train(train_loader, model, criterion, optimizer, device)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        print(optimizer.param_groups[0]['lr'])

        # valid
        # te_loss, te_acc = valid_test(dataflow, 'valid', model, device)
        te_loss, te_acc = test(test_loader, model, criterion, device)
        test_loss.append(te_loss)
        test_acc.append(te_acc)
        scheduler.step()
    print("Done!")
    # test
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
