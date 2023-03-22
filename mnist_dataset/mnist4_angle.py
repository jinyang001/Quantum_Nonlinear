import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

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


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=24,
                                               wires=list(range(self.n_wires)))

            # gates with trainable parameters

            for i in range(0, 100):
                setattr(self, 'rx' + str(i), tq.RX(has_params=True, trainable=True))
            for i in range(0, 100):
                setattr(self, 'ry' + str(i), tq.RY(has_params=True, trainable=True))
                # self.vars()['rx'+str(i)]=tq.RX(has_params=True, trainable=True)
                # self.rxgates.append(locals()['rx'+str(i)])
            for i in range(0, 100):
                setattr(self, 'crx' + str(i), tq.CRX(has_params=True, trainable=True))

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
            rx_index = 0
            ry_index = 0
            crx_index = 0

            for q in range(5):
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=0)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=1)
                rx_index += 1
                ry_index += 1
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=2)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=3)
                rx_index += 1
                ry_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[0, 1])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[1, 2])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[2, 3])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[3, 0])
                crx_index += 1


            # self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

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
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

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
        n_test_samples=200,
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
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
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
    # sss
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
