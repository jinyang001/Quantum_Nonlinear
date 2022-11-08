import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib


import matplotlib.pyplot as plt
from torchquantum.plugins import (tq2qiskit_expand_params,
                                  tq2qiskit,
                                  tq2qiskit_measurement,
                                  qiskit_assemble_circs)
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pi = np.pi
block_points_num = 100
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

train_size, test_size = int(len(data) * 0.8), len(data) - int(len(data) * 0.8)
train_dataset, test_dataset = random_split(data, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2
            self.random_layer = tq.RandomLayer(n_ops=10,
                                               wires=list(range(self.n_wires)))
            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.crx1 = tq.CRX(has_params=True, trainable=True)

            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.crx2 = tq.CRX(has_params=True, trainable=True)
            #
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.ry3 = tq.RY(has_params=True, trainable=True)
            self.crx3 = tq.CRX(has_params=True, trainable=True)
            #
            # self.rx4 = tq.RX(has_params=True, trainable=True)
            # self.ry4 = tq.RY(has_params=True, trainable=True)
            # self.crx4 = tq.CRX(has_params=True, trainable=True)

            #

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.crx0(self.q_device, wires=[0, 1])
            #
            self.rx1(self.q_device, wires=0)
            self.ry1(self.q_device, wires=1)
            self.crx1(self.q_device, wires=[0, 1])
            #
            #self.random_layer(self.q_device)
            self.rx2(self.q_device, wires=0)
            self.ry2(self.q_device, wires=1)
            self.crx2(self.q_device, wires=[0, 1])
            # #
            # self.rx3(self.q_device, wires=0)
            # self.ry3(self.q_device, wires=1)
            # self.crx3(self.q_device, wires=[0, 1])
            #
            # self.rx4(self.q_device, wires=0)
            # self.ry4(self.q_device, wires=1)
            # self.crx4(self.q_device, wires=[0, 1])

    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['2x1_u3u3'])
        # self.q_device2 = tq.QuantumDevice(n_wires=self.n_wires)

        self.q_layer = self.QLayer()
        # self.q_layer2 = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # self.measure2 = tq.MeasureAll(tq.PauliZ)


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
        # self.encoder(self.q_device2, x)
        x=x.narrow(1, 1, 1)
        # self.q_layer2(self.q_device2)
        # x = self.measure2(self.q_device2)
        #print(x.shape)
        #print(x)
        #x = x.reshape(32, 2, 1).sum(1).squeeze()
        #print(x)

        x = torch.sigmoid(x)
        #x = F.log_softmax(x, dim=1)
        return x


def train(dataloader, model, criterion, optimizer, device):
    all_losses = []
    correct = 0
    for _, (X, y) in enumerate(dataloader):
        inputs, targets = X.to(device), y.to(device)

        inputs = torch.cat((inputs, inputs), 1)

        outputs = model(inputs)
        #print(outputs)
        pred=torch.where(outputs>=0.5, 1, 0)

        #print(pred.data.view_as(pred).cpu())
        #print(targets)
        #loss = criterion(outputs, targets)
        loss = criterion(outputs, targets.reshape(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_losses.append(loss)
        #pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
        #print(correct)
    # print('inputs')
    # print(inputs)
    # print('outputs')
    # print(outputs)
    # print("pred")
    # print(pred)
    # print("targets")
    # print(targets)
    # sss
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
            inputs = torch.cat((inputs, inputs), 1)
            outputs = model(inputs)
            #test_loss += criterion(outputs, targets).item()
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
            inputs = torch.cat((inputs, inputs), 1)
            xoutput.extend(inputs.tolist())
            outputs = model(inputs)
            #test_loss += criterion(outputs, targets).item()
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
        #print(len(dataloader.dataset))

        return xoutput, youtput, toutput,test_acc


def main():
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    criterion = nn.BCELoss()
    model = QFCModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.5)
    print(model)
    print(batch_size)
    print(device)
    q_layer_circ = tq2qiskit(QFCModel().q_device, QFCModel().q_layer)

    q_layer_circ.draw(output='mpl')
    plt.show()
    epochs = 200
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n"
              f"-------------------------------")
        tr_loss, tr_acc = train(train_loader, model, criterion, optimizer, device)
        te_loss, te_acc = test(test_loader, model, criterion, device)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(te_loss)
        test_acc.append(te_acc)
    print("Done!")
    xoutput, youtput, toutput,test_acc_final = test2(test_loader, model, criterion, device)
    ####plot
    # xoutput = torch.tensor(xoutput)
    # youtput = sum(youtput, [])
    # toutput = sum(toutput, [])
    # toutput = list(map(int, toutput))
    # x2 = (xoutput.T[0]/2).tolist()
    # x = xoutput.T[0].tolist()
    # print(x)
    # print(x2)
    # print(np.cos(x2))
    # print(np.sin(x2))
    print("accuarcy")
    print(test_acc_final)
    #sss
    # x=np.cos(x2)
    # x2=np.sin(x2)



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
