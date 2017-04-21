#
# Created by Aman LaChapelle on 3/25/17.
#
# pytorch-rwa
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-EMM/LICENSE.txt
#

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class AddTask(Dataset):
    def __init__(self,
                 num_train=100000,
                 num_test=10000,
                 max_length=100,
                 path=".",
                 train=True):

        super(AddTask, self).__init__()

        self.train = train
        self.num_train = num_train
        self.num_test = num_test

        os.makedirs(path + '/data', exist_ok=True)

        # Training data - copied from the original rwa repo by Jared Ostmeyer
        #
        if not os.path.isfile(path + '/data/xs_train_{}.npy'.format(max_length)):
            values = np.random.rand(num_train, max_length, 1)
            mask = np.zeros((num_train, max_length, 1))
            for i in range(num_train):
                j1, j2 = 0, 0
                while j1 == j2:
                    j1 = np.random.randint(max_length)
                    j2 = np.random.randint(max_length)
                mask[i, j1, 0] = 1.0
                mask[i, j2, 0] = 1.0
            self.xs_train = np.concatenate((values, mask), 2)
            np.save(path + '/data/xs_train_{}.npy'.format(max_length), self.xs_train)
        else:
            self.xs_train = np.load(path + '/data/xs_train_{}.npy'.format(max_length))
        self.ys_train = np.sum(self.xs_train[:, :, 0] * self.xs_train[:, :, 1], 1)

        # Testing data - copied from the original rwa repo by Ostmeyer
        #
        if not os.path.isfile(path + '/data/xs_test_{}.npy'.format(max_length)):
            values = np.random.rand(num_test, max_length, 1)
            mask = np.zeros((num_test, max_length, 1))
            for i in range(num_test):
                j1, j2 = 0, 0
                while j1 == j2:
                    j1 = np.random.randint(max_length)
                    j2 = np.random.randint(max_length)
                mask[i, j1, 0] = 1.0
                mask[i, j2, 0] = 1.0
            self.xs_test = np.concatenate((values, mask), 2)
            np.save(path + '/data/xs_test_{}.npy'.format(max_length), self.xs_test)
        else:
            self.xs_test = np.load(path + '/data/xs_test_{}.npy'.format(max_length))
        self.ys_test = np.sum(self.xs_test[:, :, 0] * self.xs_test[:, :, 1], 1)

    def __getitem__(self, item):
        if self.train:
            x_sample = torch.from_numpy(self.xs_train[item, :, :]).float()
            y_sample = torch.FloatTensor([self.ys_train[item]])
        else:
            x_sample = torch.from_numpy(self.xs_test[item, :, :]).float()
            y_sample = torch.FloatTensor([self.ys_test[item]])

        return x_sample, y_sample

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_test


class CopyTask(Dataset):
    def __init__(self,
                 num_train=100000,
                 max_length=100,
                 input_size=(1, 9)):

        super(CopyTask, self).__init__()

        self.num_train = num_train
        self.max_length = max_length
        self.input_size = input_size

        self.input_tensor = torch.FloatTensor(*input_size).uniform_(0, 1)

    def __getitem__(self, index):
        sample = []
        sample_label = []

        rand_seq_len = np.random.randint(low=int(self.max_length-self.max_length/5), high=self.max_length)
        zeros = torch.zeros(*self.input_size)

        for i in range(rand_seq_len):
            sample.append(torch.bernoulli(self.input_tensor))
            sample_label.append(zeros)

        sample.append(torch.ones(*zeros.size()) * 0.5)
        sample_label.append(zeros)

        for i in range(rand_seq_len):
            sample_label.append(sample[i])
            sample.append(zeros)

        sample = torch.stack(sample).view(2 * rand_seq_len + 1, *self.input_size)
        sample_label = torch.cat(sample_label).view(2 * rand_seq_len + 1, *self.input_size)

        return sample, sample_label

    def __len__(self):
        return self.num_train


if __name__ == "__main__":
    add = CopyTask()
    x, y = add[0]
    print(x, y)
