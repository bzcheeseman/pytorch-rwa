#
# Created by Aman LaChapelle on 4/20/17.
#
# pytorch-rwa
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-rwa/LICENSE.txt
#

from rwa_model import RWAGPU
from utils import AddTask

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value

configure("training/gpu_test_bn_7_cuda")

num_features = 2
num_classes = 1
num_filters = 10  # 15 works, 10 works at lr = 0.001
                  # 10 works at lr = 0.0015, 0.002, 0.003
                  # 15 works at lr = 0.002, little slower than the other way
kernel_width = 1
# num_cells = 250
batch = 100
# rwa = RWA(num_features, num_cells, num_classes, fwd_type="cumulative")
rwa = RWAGPU(num_features, kernel_width, num_filters, num_classes)

criterion = nn.MSELoss()

print_steps = 5

current_lr = 0.003

running_loss = 0.0
time_since_decay = 0
accumulated_loss = []

test = AddTask(100000, 10000, 100)

data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

s, n, d, h, a_max = rwa.init_sndha(batch)
rwa.register_parameter('s', s)  # make sure s changes after each optimizer step

optimizer = optim.Adam(rwa.parameters(), lr=current_lr)

rwa.train()
rwa.cuda()

for epoch in range(5):

    for i, data in enumerate(data_loader, 0):

        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs, rwa.s, n_new, d_new, h_new, a_newmax = rwa(inputs.cuda(), rwa.s, n.cuda(), d.cuda(), h.cuda(), a_max.cuda())

        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        n = Variable(n_new.data)
        d = Variable(d_new.data)
        h = Variable(h_new.data)
        a_max = Variable(a_newmax.data)

        running_loss += loss.data[0]
        accumulated_loss.append(loss.data[0])
        time_since_decay += 1

        if i % print_steps == print_steps - 1:
            current_step = i + 1 + len(data_loader) * epoch
            print("Current step: ", current_step, "Loss: ", running_loss / print_steps)
            log_value("Loss", running_loss / print_steps, step=current_step)
            #             log_value("LR", current_lr, step=current_step)
            log_value("Error", np.abs(outputs.data[0, 0] - labels.data[0, 0]), step=current_step)
            log_value("Output", outputs.data[0, 0], step=current_step)

            # if time_since_decay >= int(0.5 * (1. / current_lr)):
            #     if np.abs(np.mean(np.diff(accumulated_loss))) <= current_lr:
            #         torch.save(rwa.state_dict(), "models/add.dat")
            #         current_lr = max([current_lr * 1e-1, 1e-8])
            #         print("lr decayed to: ", current_lr)
            #         optimizer = optim.Adam(rwa.parameters(), lr=current_lr)
            #         accumulated_loss.clear()
            #         time_since_decay = 0

            running_loss = 0.0

torch.save(rwa.state_dict(), "models/rwa_add.dat")
print("Finished Training")