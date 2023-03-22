import qiskit
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torchquantum import C_DTYPE, switch_little_big_endian_state

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
from qiskit import IBMQ, QuantumCircuit

from torchquantum.plugins.qiskit_plugin import tq2qiskit_initialize


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 12
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

            for q in range(0, self.n_wires, 2):
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q + 1)
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q, q + 1])
                crx_index += 1
                rx_index += 1
                ry_index += 1
            for q in range(0, self.n_wires, 2):
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q + 1)
                rx_index += 1
                ry_index += 1

            for q in range(0, self.n_wires, 4):
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q, q + 1])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q + 1, q + 2])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q + 2, q + 3])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q + 3, q + 2])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q + 2, q + 1])
                crx_index += 1
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q + 1, q])
                crx_index += 1

            for q in range(0, self.n_wires, 2):
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q + 1)
                rx_index += 1
                ry_index += 1
            for q in range(11):
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q, q + 1])
                crx_index += 1

            for q in range(11, 0,-1):
                getattr(self, 'crx' + str(crx_index))(self.q_device, wires=[q, q -1])
                crx_index += 1

    def __init__(self):
        super().__init__()
        self.n_wires = 12
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        if use_qiskit:
            # test initialize
            circ_all = []
            for k in range(bsz):
                circ = QuantumCircuit(self.n_wires)
                state = x[k]
                block1 = torch.stack((state[0], state[1], state[4], state[5]))
                block2 = torch.stack((state[2], state[3], state[6], state[7]))
                block3 = torch.stack((state[8], state[9], state[12], state[13]))
                block4 = torch.stack((state[10], state[11], state[14], state[15]))
                chunks1 = [block1, block2, block3, block4]

                for i in range(len(chunks1)):
                    chunks1[i] = chunks1[i] / (torch.sqrt((chunks1[i].abs() ** 2).sum(dim=-1))).unsqueeze(-1)
                    # chunks1[i] = chunks1[i].numpy()
                    chunks1[i] = np.complex128(chunks1[i])
                    chunks1[i] = chunks1[i] / (np.absolute(chunks1[i]) ** 2).sum()
                    chunks1[i] = switch_little_big_endian_state(chunks1[i])
                    # chunks1[i] = chunks1[i] / np.linalg.norm(chunks1[i])
                    # chunks1[i] = np.complex128(chunks1[i])

                qiskit.circuit.library.data_preparation.state_preparation._EPS = 1e-5
                circ.initialize(chunks1[0], [0, 1])
                circ.initialize(chunks1[1], [2, 3])
                circ.initialize(chunks1[2], [4, 5])
                circ.initialize(chunks1[3], [6, 7])
                circ_all.append(circ)

            encoder_circs = circ_all
            q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
            measurement_circ = tq2qiskit_measurement(self.q_device,
                                                     self.measure)
            assembled_circs = qiskit_assemble_circs(encoder_circs,
                                                    q_layer_circ,
                                                    measurement_circ)

            x0 = self.qiskit_processor.process_ready_circs(
                self.q_device, assembled_circs).to(devi)

            x = x0
            # encoder_circs = tq2qiskit_initialize(self.q_device, self.q_device.get_states_1d().to(torch.complex64).detach().cpu().numpy())
            # # encoder_circs = tq2qiskit_expand_params(self.q_device, x,
            # #                                         self.encoder.func_list)
            # q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
            # measurement_circ = tq2qiskit_measurement(self.q_device,
            #                                          self.measure)
            # assembled_circs = qiskit_assemble_circs(encoder_circs,
            #                                         q_layer_circ,
            #                                         measurement_circ)
            #
            # x0 = self.qiskit_processor.process_ready_circs(
            #     self.q_device, assembled_circs).to(devi)
            #
            # x = x0

        else:
            out = x
            out2 = torch.zeros(
                x.shape[0], 2 ** self.q_device.n_wires,
                device=x.device)

            for i, t in enumerate(out):
                # chunks = [t[x:x + 4] for x in range(0, len(t), 4)]
                block1 = torch.stack((t[0], t[1], t[4], t[5]))
                block2 = torch.stack((t[0], t[1], t[4], t[5]))
                block3 = torch.stack((t[2], t[3], t[6], t[7]))
                block4 = torch.stack((t[2], t[3], t[6], t[7]))
                block5 = torch.stack((t[8], t[9], t[12], t[13]))
                block6= torch.stack((t[10], t[11], t[14], t[15]))

                chunks1 = [block1, block2, block3, block4,block5, block6]

                temp = torch.tensor(chunks1[0])
                temp = temp / (torch.sqrt((temp.abs() ** 2).sum(dim=-1))).unsqueeze(-1)
                for c in range(1, len(chunks1)):
                    temp2 = torch.tensor(chunks1[c])
                    temp2 = temp2 / (torch.sqrt((temp2.abs() ** 2).sum(dim=-1))).unsqueeze(-1)
                    temp = torch.kron(temp, temp2)

                out2[i] = temp
                # print(temp.shape)
                # print(temp)
                # sss
            states1d = out2
            states1d = states1d.view([out2.shape[0]] + [2] * self.q_device.n_wires)
            self.q_device.states = states1d.type(C_DTYPE)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 6).sum(-1).squeeze()
        # print(x.shape)
        # print(x)
        # sss
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
        # print(loss)
        # ss
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

    return train_loss, train_acc


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)

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
    # print(output_all)
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=20,
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
        # digits_of_interest=[1, 3, 6, 9],
        digits_of_interest=[3, 6],
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
    #     backend_name = 'ibmq_qasm_simulator'
    #     processor_simulation = QiskitProcessor(use_real_qc=True, backend_name=backend_name,optimization_level=0, hub='ibm-q-lanl',
    #                                            group='lanl', project='quantum-optimiza')
    #     model.set_qiskit_processor(processor_simulation)
    #     valid_test(dataflow, 'test', model, device, qiskit=True)
    #
    #     # then try to run on REAL QC
    #     backend_name = 'ibm_cairo'
    #     print(f"\nTest on Real Quantum Computer {backend_name}")
    #     # Please specify your own hub group and project if you have the
    #     # IBMQ premium plan to access more machines.
    #     processor_real_qc = QiskitProcessor(use_real_qc=True,
    #                                         backend_name=backend_name,
    #                                         # optimization_level=0,
    #                                         initial_layout=[0, 1, 4, 7, 10, 12, 13, 14],
    #                                         hub='ibm-q-lanl',
    #                                         group='lanl',
    #                                         project='quantum-optimiza',
    #                                         )
    #     model.set_qiskit_processor(processor_real_qc)
    #     valid_test(dataflow, 'test', model, device, qiskit=True)
    # except ImportError:
    #     print("Please install qiskit, create an IBM Q Experience Account and "
    #           "save the account token according to the instruction at "
    #           "'https://github.com/Qiskit/qiskit-ibmq-provider', "
    #           "then try again.")


if __name__ == '__main__':
    main()
