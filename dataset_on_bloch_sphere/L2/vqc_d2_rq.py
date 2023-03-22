import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchquantum import C_DTYPE

import torchquantum as tq
import torchquantum.functional as tqf

from torchquantum.plugins import (tq2qiskit_expand_params,
                                  tq2qiskit,
                                  tq2qiskit_measurement,
                                  qiskit_assemble_circs)

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qiskit import IBMQ
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

train_size, test_size, test_size2 = int(len(data) * 0.8), int(len(data) * 0.1), len(data) - int(len(data) * 0.8)- int(len(data) * 0.1)
train_dataset, test_dataset,test_dataset2 = random_split(data, [train_size, test_size,test_size2])

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader2 = DataLoader(dataset=test_dataset2, batch_size=len(test_dataset2), shuffle=False, num_workers=0, drop_last=False)
# print(len(test_dataset2))
# sss
# for batch, (X, y) in enumerate(train_loader):
#     print("batch:", batch)
#     print(f'X.shape:{X.shape}, y.shape:{y.shape}')
#     Data, Label = X, y
#     print("data:", Data)
#     print("Label:", Label)
#     break
# sss
class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1

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
            self.rz3 = tq.RZ(has_params=True, trainable=True)
            self.crx3 = tq.CRX(has_params=True, trainable=True)

            self.crx4 = tq.CRX(has_params=True, trainable=True)
            self.crx5 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=0)
            self.rz0(self.q_device, wires=0)
            # self.crx0(self.q_device, wires=[0, 1])
            self.rx1(self.q_device, wires=0)
            self.ry1(self.q_device, wires=0)
            self.rz1(self.q_device, wires=0)
            # self.crx1(self.q_device, wires=[2, 3])
            #
            self.rx2(self.q_device, wires=0)
            self.ry2(self.q_device, wires=0)
            self.rz2(self.q_device, wires=0)

    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['1x1_u3'])
        self.q_layer = self.QLayer()

        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x, use_qiskit=False):
        # bsz = x.shape[0]

        devi = x.device
        if use_qiskit:
            encoder_circs = tq2qiskit_expand_params(self.q_device, x,
                                                    self.encoder.func_list)
            q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
            measurement_circ = tq2qiskit_measurement(self.q_device,
                                                     self.measure)
            assembled_circs = qiskit_assemble_circs(encoder_circs,
                                                    q_layer_circ,
                                                    measurement_circ)
            x0 = self.qiskit_processor.process_ready_circs(
                self.q_device, assembled_circs).to(devi)
            # x1 = self.qiskit_processor.process_parameterized(
            #     self.q_device, self.encoder, self.q_layer, self.measure, x)
            # print((x0-x1).max())
            x = x0

        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)
        x = torch.sigmoid(x)
        # x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        # x = F.log_softmax(x, dim=1)

        return x


def train(dataloader, model, criterion, optimizer, device):
    all_losses = []
    correct = 0
    for _, (X, y) in enumerate(dataloader):
        inputs, targets = X.to(device), y.to(device)


        outputs = model(inputs)

        pred=torch.where(outputs>=0.5, 1, 0)

        loss = criterion(outputs, targets.reshape(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_losses.append(loss)

        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()


    train_acc = 100. * correct / len(dataloader.dataset)
    train_loss = float(sum(all_losses)) / len(dataloader.dataset)
    return train_loss, train_acc


def test(dataloader, model, criterion, device, qiskit=False):
    # model.eval()
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

        print(f"{'valid'} set accuracy: {test_acc}")
        print(f"{'valid'} set loss: {test_loss}")
        return test_loss, test_acc


def test2(dataloader, model, criterion, device, qiskit=False):
    # model.eval()
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
            outputs = model(inputs, use_qiskit=qiskit)
            test_loss += criterion(outputs.float(), targets.reshape(-1, 1).float()).item()
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
        print(f"{'test'} set accuracy: {test_acc}")
        print(f"{'test'} set loss: {test_loss}")

        return xoutput, youtput, toutput,test_acc


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

    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    #
    # dataset = MNIST(
    #     root='./mnist_data',
    #     train_valid_split_ratio=[0.9, 0.1],
    #     digits_of_interest=[3, 6],
    #     n_test_samples=200,
    # )
    # dataflow = dict()
    #
    # for split in dataset:
    #     sampler = torch.utils.data.RandomSampler(dataset[split])
    #     dataflow[split] = torch.utils.data.DataLoader(
    #         dataset[split],
    #         batch_size=256,
    #         sampler=sampler,
    #         num_workers=8,
    #         pin_memory=True)

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

    # plt.figure(figsize=(5, 3))
    # plt.plot(range(1, epochs + 1), train_acc)
    # plt.plot(range(1, epochs + 1), test_acc)
    # plt.title('Accuracy in {} epochs'.format(epochs))
    #
    # plt.figure(figsize=(5, 3))
    # plt.plot(range(1, epochs + 1), train_loss)
    # plt.plot(range(1, epochs + 1), test_loss)
    # plt.title('Loss in {} epochs'.format(epochs))
    # plt.show()

    # run on Qiskit simulator and real Quantum Computers
    # try:
    #     from qiskit import IBMQ
    #     from torchquantum.plugins import QiskitProcessor
    #
    #     # firstly perform simulate
    #     # print(f"\nTest with Qiskit Simulator")
    #     # backend_name = 'ibmq_qasm_simulator'
    #     # processor_simulation = QiskitProcessor(use_real_qc=True, backend_name=backend_name, optimization_level=0,
    #     #                                        hub='ibm-q-lanl',
    #     #                                        group='lanl', project='quantum-optimiza')
    #     # model.set_qiskit_processor(processor_simulation)
    #     # # valid_test(dataflow, 'test', model, device, qiskit=True)
    #     # test2(test_loader2, model, criterion, device, qiskit=True)
    #     # then try to run on REAL QC
    #     backend_name = 'ibmq_jakarta'
    #     print(f"\nTest on Real Quantum Computer {backend_name}")
    #     # Please specify your own hub group and project if you have the
    #     # IBMQ premium plan to access more machines.
    #     processor_real_qc = QiskitProcessor(use_real_qc=True,
    #                                         backend_name=backend_name,
    #                                         # optimization_level=0,
    #                                         # initial_layout=[3, 5, 8, 11, 14, 16, 19, 22],
    #                                         hub='ibm-q-lanl',
    #                                         group='lanl',
    #                                         project='quantum-optimiza',
    #                                         )
    #     model.set_qiskit_processor(processor_real_qc)
    #     test2(test_loader2, model, criterion, device, qiskit=True)
    # except ImportError:
    #     print("Please install qiskit, create an IBM Q Experience Account and "
    #           "save the account token according to the instruction at "
    #           "'https://github.com/Qiskit/qiskit-ibmq-provider', "
    #           "then try again.")


if __name__ == '__main__':
    main()


