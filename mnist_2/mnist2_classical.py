import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
        #                        kernel_size=3, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
        #                        kernel_size=2, stride=1, padding=0)
        # self.linear1 = nn.Linear(16, 8)
        # self.linear2 = nn.Linear(8, 4)
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
        #                        kernel_size=3, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=6,
        #                        kernel_size=2, stride=1, padding=0)
        self.linear1 = nn.Linear(16, 2)
        # self.linear2 = nn.Linear(8, 4)
        self.non_linear = nn.Tanh()
        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        # x = F.avg_pool2d(x, 6)

        # x = self.conv1(x)
        #
        # x = self.non_linear(x)
        # x = self.conv2(x)
        #
        # x = self.non_linear(x)
        # # print(x.shape)
        # # ss
        # x = x.reshape(x.shape[0], -1)

        x = self.linear1(x)
        # x = self.non_linear(x)
        # x = self.linear2(x)
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

    return train_loss, train_acc


def valid_test(dataflow, split, model, device):
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
    model = CNN().to(device)

    net = CNN()
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    # sss
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
    valid_test(dataflow, 'test', model, device)
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



if __name__ == '__main__':
    main()

