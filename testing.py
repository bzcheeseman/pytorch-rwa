#
# Created by Aman LaChapelle on 4/20/17.
#
# pytorch-rwa
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-rwa/LICENSE.txt
#

from model import RWAGPU, RWA, CGRURWA
from utils import AddTask, CopyTask

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value

configure("training/cgru_11")
num_features = 2
num_classes = 1
num_filters = 250
kernel_width = 1
num_cells = 250
batch = 100
# rwa = RWA(num_features, num_cells, num_classes, decay=True, fwd_type="cumulative")
rwa = CGRURWA(num_features, 10, num_filters, num_classes)
# this seems slower if time_steps is much smaller than the sequence
# it also learns the 1000 add task pretty fast, step ~1600, 100 add task in ~450 steps
rwa.cuda()

criterion = nn.MSELoss()

print_steps = 10

current_lr = 0.001

running_loss = 0.0
time_since_decay = 0
accumulated_loss = []

test = AddTask(100000, 10000, 100)

data_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

hidden = rwa.init_hidden(batch)
# rwa.load_state_dict(torch.load("models/rwa_add.dat"))

optimizer = optim.Adam(rwa.parameters(), lr=current_lr)  # add weight decay?

rwa.train()

for epoch in range(5):

    for i, data in enumerate(data_loader, 0):

        inputs, labels = data
        inputs = Variable(inputs).cuda(async=True)
        labels = Variable(labels).cuda(async=True)

        optimizer.zero_grad()
        outputs, hidden = rwa(inputs, hidden)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        hidden = rwa.detach_hidden(hidden)

        running_loss += loss.data[0]
        accumulated_loss.append(loss.data[0])
        time_since_decay += 1

        if i % print_steps == print_steps - 1:
            current_step = i + 1 + len(data_loader) * epoch
            print("Current step: ", current_step, "Loss: ", running_loss / print_steps)
            log_value("Loss", running_loss / print_steps, step=current_step)
            log_value("Outputs", outputs.cpu().data.numpy()[0], step=current_step)
            log_value("Error", (outputs - labels).cpu().data.numpy()[0], step=current_step)

            # if time_since_decay >= int(0.8 * (1. / current_lr)):
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

test = AddTask(50, 200)
testloader = DataLoader(test, batch_size=batch)

for i, data in enumerate(testloader, 0):
    inputs, label = data
    outputs, hidden = rwa(Variable(inputs.cuda()), hidden)
    print(criterion(outputs, Variable(label).cuda()))
    # print(label - outputs.cpu().data)
    # plt.imshow(outputs.cpu().squeeze().data.numpy())
    # plt.show()
    # plt.imshow(label.cpu().squeeze().numpy())
    # plt.show()


