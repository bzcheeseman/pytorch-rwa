#
# Created by Aman LaChapelle on 4/11/17.
#
# pytorch-rwa
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-rwa/LICENSE.txt
#

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class Trainer:
    def __init__(self,
                 net,
                 criterion,
                 optimizer,
                 data_loader,
                 log_steps=25,
                 max_epochs=-1,
                 log_dir="./log_data"):

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.log_steps = log_steps
        self.terminate_criterion = max_epochs if max_epochs >= 0 else False

        use_tensorboard = True
        try:
            from tensorboard_logger import configure, log_value
        except ImportError:
            use_tensorboard = False
            configure = None
            log_value = None

        if use_tensorboard:
            configure(log_dir)
        else:
            self.log_dir = log_dir

        self.use_tensorboard = use_tensorboard

    def _train_epoch_termination(self, starting_lr, ending_lr, *net_args):
        current_lr = starting_lr
        for epoch in range(self.terminate_criterion):
            running_loss = 0.0
            accumulated_loss = []

            for i, data in enumerate(self.data_loader, 0):

                inputs, labels = data
                inputs = Variable(inputs)
                labels = Variable(labels)

                self.optimizer.zero_grad()
                outputs, *net_args = self.net(inputs, *net_args)

                for arg in net_args:
                    arg = Variable(arg.data)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.data[0]
                accumulated_loss.append(loss.data[0])

                if i % self.log_steps == self.log_steps - 1:
                    if self.use_tensorboard:
                        current_step = i + 1 + len(self.data_loader) * epoch
                        log_value(running_loss/self.log_steps, step=current_step)

                    print('[epoch: %d, i: %5d] average loss: %.3f' % (epoch + 1, i + 1,
                                                                      running_loss / self.log_steps))

                    if np.mean(np.abs(np.diff(accumulated_loss))) <= 0.2 * current_lr:
                        torch.save(self.net.state_dict(), "models/add.dat")
                        current_lr = max([current_lr * 1e-1, ending_lr])
                        print("lr decayed to: ", current_lr)
                        self.optimizer = optim.Adam(self.net.parameters(), lr=current_lr)
                        accumulated_loss.clear()

                    running_loss = 0.0

        torch.save(self.net.state_dict(), "models/rwa_add.dat")
        print("Finished Training")

    def _train_lr_termination(self, starting_lr, ending_lr, *net_args):
        current_lr = starting_lr

        current_step = 0
        while current_lr >= ending_lr - 1e-15:
            running_loss = 0.0
            accumulated_loss = []

            for i, data in enumerate(self.data_loader, 0):
                current_step += 1

                inputs, labels = data
                inputs = Variable(inputs)
                labels = Variable(labels)

                self.optimizer.zero_grad()
                outputs, *net_args = self.net(inputs, *net_args)

                for arg in net_args:
                    arg = Variable(arg.data)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.data[0]
                accumulated_loss.append(loss.data[0])

                if i % self.log_steps == self.log_steps - 1:
                    if self.use_tensorboard:
                        log_value(running_loss/self.log_steps, step=current_step)

                    print('[lr: %.3e, i: %5d] average loss: %.3f' % (current_lr, i + 1,
                                                                      running_loss / self.log_steps))

                    if np.mean(np.abs(np.diff(accumulated_loss))) <= 0.2 * current_lr:
                        torch.save(self.net.state_dict(), "models/add.dat")

                        current_lr = max([current_lr * 1e-1, ending_lr])

                        print("lr decayed to: ", current_lr)
                        self.optimizer = optim.Adam(self.net.parameters(), lr=current_lr)
                        accumulated_loss.clear()

                    running_loss = 0.0

    def train(self, starting_lr, ending_lr, *net_args):
        if not self.terminate_criterion:
            return self._train_lr_termination(starting_lr, ending_lr, *net_args)
        else:
            return self._train_epoch_termination(starting_lr, ending_lr, *net_args)
