import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

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


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 6
            self.random_layer = tq.RandomLayer(n_ops=24,
                                               wires=list(range(self.n_wires)))

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.crx1 = tq.CRX(has_params=True, trainable=True)
            self.crx2 = tq.CRX(has_params=True, trainable=True)
            self.crx3 = tq.CRX(has_params=True, trainable=True)
            self.crx4 = tq.CRX(has_params=True, trainable=True)
            self.crx5 = tq.CRX(has_params=True, trainable=True)

            self.rx00 = tq.RX(has_params=True, trainable=True)
            self.ry00 = tq.RY(has_params=True, trainable=True)
            self.rx11 = tq.RX(has_params=True, trainable=True)
            self.ry11 = tq.RY(has_params=True, trainable=True)
            self.rx22 = tq.RX(has_params=True, trainable=True)
            self.ry22 = tq.RY(has_params=True, trainable=True)

            self.crx00 = tq.CRX(has_params=True, trainable=True)
            self.crx11 = tq.CRX(has_params=True, trainable=True)
            self.crx22 = tq.CRX(has_params=True, trainable=True)
            self.crx33 = tq.CRX(has_params=True, trainable=True)
            self.crx44 = tq.CRX(has_params=True, trainable=True)
            self.crx55 = tq.CRX(has_params=True, trainable=True)

            # self.rx00 = tq.RX(has_params=True, trainable=True)
            # self.ry00 = tq.RY(has_params=True, trainable=True)
            # self.rx11 = tq.RX(has_params=True, trainable=True)
            # self.ry11 = tq.RY(has_params=True, trainable=True)
            # self.crx00 = tq.CRX(has_params=True, trainable=True)
            # self.crx11 = tq.CRX(has_params=True, trainable=True)
            # self.crx22 = tq.CRX(has_params=True, trainable=True)
            # self.crx33 = tq.CRX(has_params=True, trainable=True)
            #
            # self.rx000 = tq.RX(has_params=True, trainable=True)
            # self.ry000 = tq.RY(has_params=True, trainable=True)
            # self.rx111 = tq.RX(has_params=True, trainable=True)
            # self.ry111 = tq.RY(has_params=True, trainable=True)
            # self.crx000 = tq.CRX(has_params=True, trainable=True)
            # self.crx111 = tq.CRX(has_params=True, trainable=True)
            # self.crx222 = tq.CRX(has_params=True, trainable=True)
            # self.crx333 = tq.CRX(has_params=True, trainable=True)
            #
            # self.rx0000 = tq.RX(has_params=True, trainable=True)
            # self.ry0000 = tq.RY(has_params=True, trainable=True)
            # self.rx1111 = tq.RX(has_params=True, trainable=True)
            # self.ry1111 = tq.RY(has_params=True, trainable=True)
            # self.crx0000 = tq.CRX(has_params=True, trainable=True)
            # self.crx1111 = tq.CRX(has_params=True, trainable=True)
            # self.crx2222 = tq.CRX(has_params=True, trainable=True)
            # self.crx3333 = tq.CRX(has_params=True, trainable=True)


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

            #self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.rx1(self.q_device, wires=2)
            self.ry1(self.q_device, wires=3)
            self.rx2(self.q_device, wires=4)
            self.ry2(self.q_device, wires=5)

            self.crx0(self.q_device, wires=[0, 1])
            self.crx1(self.q_device, wires=[1, 2])
            self.crx2(self.q_device, wires=[2, 0])

            self.crx3(self.q_device, wires=[3, 4])
            self.crx4(self.q_device, wires=[4, 5])
            self.crx5(self.q_device, wires=[5, 3])

            self.rx00(self.q_device, wires=0)
            self.ry00(self.q_device, wires=1)
            self.rx11(self.q_device, wires=2)
            self.ry11(self.q_device, wires=3)
            self.rx22(self.q_device, wires=4)
            self.ry22(self.q_device, wires=5)

            self.crx00(self.q_device, wires=[0, 1])
            self.crx11(self.q_device, wires=[2, 3])
            self.crx22(self.q_device, wires=[1, 2])
            self.crx33(self.q_device, wires=[4, 5])
            self.crx44(self.q_device, wires=[3, 4])
            self.crx55(self.q_device, wires=[5, 0])

            # self.rx00(self.q_device, wires=0)
            # self.ry00(self.q_device, wires=1)
            # self.rx11(self.q_device, wires=2)
            # self.ry11(self.q_device, wires=3)
            # self.crx00(self.q_device, wires=[0, 1])
            # self.crx22(self.q_device, wires=[2, 3])
            # self.crx11(self.q_device, wires=[1, 2])
            # self.crx33(self.q_device, wires=[3, 0])
            #
            # self.rx000(self.q_device, wires=0)
            # self.ry000(self.q_device, wires=1)
            # self.rx111(self.q_device, wires=2)
            # self.ry111(self.q_device, wires=3)
            # self.crx000(self.q_device, wires=[0, 1])
            # self.crx222(self.q_device, wires=[2, 3])
            # self.crx111(self.q_device, wires=[1, 2])
            # self.crx333(self.q_device, wires=[3, 0])
            #
            # self.rx0000(self.q_device, wires=0)
            # self.ry0000(self.q_device, wires=1)
            # self.rx1111(self.q_device, wires=2)
            # self.ry1111(self.q_device, wires=3)
            # self.crx0000(self.q_device, wires=[0, 1])
            # self.crx2222(self.q_device, wires=[2, 3])
            # self.crx1111(self.q_device, wires=[1, 2])
            # self.crx3333(self.q_device, wires=[3, 0])

            #self.random_layer(self.q_device)
            # add some more non-parameterized gates (add on-the-fly)
            # tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
            #              parent_graph=self.graph)
            # tqf.sx(self.q_device, wires=2, static=self.static_mode,
            #        parent_graph=self.graph)
            # tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
            #          parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 6
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    # def forward(self, x):
    #     bsz = x.shape[0]
    #     x = F.avg_pool2d(x, 6).view(bsz, 16)
    #     print(x[0])
    #     self.encoder(self.q_device, x)
    #     print(self.q_device.get_states_1d()[0])
    #     self.q_layer(self.q_device)
    #     # encoder_circs = tq2qiskit_expand_params(self.q_device, x,
    #     #                                         self.encoder.func_list)
    #     q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
    #     measurement_circ = tq2qiskit_measurement(self.q_device,
    #                                              self.measure)
    #     # assembled_circs = qiskit_assemble_circs(encoder_circs,
    #     #                                         q_layer_circ,
    #     #                                         measurement_circ)
    #     #print(measurement_circ)
    #     #print(q_layer_circ[0].draw())
    #     q_layer_circ.draw(output='mpl')
    #     plt.show()
    #
    #     ssss
    #     x = self.measure(self.q_device)
    #
    #     x = F.log_softmax(x, dim=1)
    #
    #     return x

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        # print(x.shape)
        # print(x)
        # x = torch.cat((x, x), 1)
        out = x.tolist()

        for i, t in enumerate(out):
            chunks = [t[x:x + 4] for x in range(0, len(t), 4)]
            # print(chunks)
            # print(chunks[0])
            # print(chunks[1])
            # print(chunks[2])
            # print(chunks[3])
            #
            # sss
            # print(chunks)
            indx = [0, 1]
            chunks1 = [chunks[_ind] for _ind in indx]
            # chunks2 = random.shuffle(chunks)
            # print(chunks1)
            # print(chunks2)
            # sss
            temp = chunks1[0]
            for c in range(1, len(chunks1)):
                temp = np.kron(temp, chunks1[c])
            out[i] = temp
        x = torch.tensor(out).to('cuda')
        # print(x.shape)
        # print(x)
        # ss
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        # print(x)
        # x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        # print(x)

        x = F.log_softmax(x, dim=1)

        return x


def train(dataflow, model, device, optimizer):
    # all_losses = []
    target_all = []
    output_all = []
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)
        # print('targets')
        # print(targets)
        # print(targets.shape)
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # all_losses.append(loss.item())
        target_all.append(targets)
        output_all.append(outputs)
        print(f"loss: {loss.item()}", end='\r')

    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    train_acc = 100. * corrects / size
    train_loss = F.nll_loss(output_all, target_all).item()
    # train_loss = float(np.mean(all_losses))
    # print(size)
    # print(train_acc)
    # print(train_loss)
    return train_loss, train_acc


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    # print((output_all))
    # print(target_all)
    # print(target_all.view(-1, 1))
    # print(indices)

    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    # print(masks)
    # sss
    size = target_all.shape[0]
    corrects = masks.sum().item()
    test_acc = 100. * corrects / size
    test_loss = F.nll_loss(output_all, target_all).item()
    # print('targets')
    # print(target_all)
    print(f"{split} set accuracy: {test_acc}")
    print(f"{split} set loss: {test_loss}")
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[1, 3, 6, 9],
        n_test_samples=75,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
    epochs = 20
    print(device)
    q_layer_circ = tq2qiskit(QFCModel().q_device, QFCModel().q_layer)

    q_layer_circ.draw(output='mpl')
    plt.show()

    print('start')
    for epoch in range(epochs):
        # train
        print(f"Epoch {epoch}:")
        tr_loss, tr_acc = train(dataflow, model, device, optimizer)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        print(optimizer.param_groups[0]['lr'])

        # valid
        te_loss, te_acc = valid_test(dataflow, 'valid', model, device)
        test_loss.append(te_loss)
        test_acc.append(te_acc)
        scheduler.step()
    print("Done!")
    # test
    valid_test(dataflow, 'test', model, device, qiskit=False)
    # graph
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

    # run on Qiskit simulator and real Quantum Computers
    # try:
    #     from qiskit import IBMQ
    #     from torchquantum.plugins import QiskitProcessor
    #
    #     # firstly perform simulate
    #     print(f"\nTest with Qiskit Simulator")
    #     processor_simulation = QiskitProcessor(use_real_qc=False)
    #     model.set_qiskit_processor(processor_simulation)
    #     valid_test(dataflow, 'test', model, device, qiskit=True)
    # #
    # #     # then try to run on REAL QC
    # #     backend_name = 'ibmq_lima'
    # #     print(f"\nTest on Real Quantum Computer {backend_name}")
    # #     # Please specify your own hub group and project if you have the
    # #     # IBMQ premium plan to access more machines.
    # #     processor_real_qc = QiskitProcessor(use_real_qc=True,
    # #                                         backend_name=backend_name,
    # #                                         hub='ibm-q',
    # #                                         group='open',
    # #                                         project='main',
    # #                                         )
    # #     model.set_qiskit_processor(processor_real_qc)
    # #     valid_test(dataflow, 'test', model, device, qiskit=True)
    # except ImportError:
    #     print("Please install qiskit, create an IBM Q Experience Account and "
    #           "save the account token according to the instruction at "
    #           "'https://github.com/Qiskit/qiskit-ibmq-provider', "
    #           "then try again.")


if __name__ == '__main__':
    main()
