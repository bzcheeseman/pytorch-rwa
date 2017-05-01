#
# Created by Aman LaChapelle on 4/11/17.
#
# pytorch-rwa
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-rwa/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct
import numpy as np


class RWACell(nn.Module):
    def __init__(self,
                 num_features,
                 num_cells,
                 decay=True,
                 init=1.0,
                 activation=Funct.tanh):
        super(RWACell, self).__init__()

        self.num_features = num_features
        self.num_cells = num_cells
        self.activation = activation
        self.decay = decay
        self.init = init

        self._gpu = False

        #### g, a, and u gates initialization #####
        ga_init_factor = np.sqrt((6.0 * init) / (num_features + 2.0 * num_cells))
        u_init_factor = np.sqrt((6.0 * init) / (num_features + num_cells))

        self.g = nn.Linear(self.num_features + self.num_cells, self.num_cells)
        self.g.weight.data.uniform_(-ga_init_factor, ga_init_factor)
        self.g.bias.data.zero_()

        self.u = nn.Linear(self.num_features, self.num_cells)
        self.u.weight.data.uniform_(-u_init_factor, u_init_factor)
        self.u.bias.data.zero_()

        self.a = nn.Linear(self.num_features + self.num_cells, self.num_cells, bias=False)
        self.a.weight.data.uniform_(-ga_init_factor, ga_init_factor)

        #### Attention/decay mechanism #####
        self.decay = nn.Linear(self.num_features + self.num_cells, self.num_cells, bias=False)
        self.decay.weight.data.uniform_(-ga_init_factor, ga_init_factor)

    def cuda(self, device_id=None):
        self._gpu = True
        super(RWACell, self).cuda(device_id)

    def init_hidden(self, batch_size):
        typ = torch.FloatTensor
        if self._gpu:
            typ = torch.cuda.FloatTensor
        s = nn.Parameter(typ(batch_size, self.num_cells).normal_(0.0, self.init), requires_grad=True)
        self.register_parameter('s', s)
        n = Variable(typ(batch_size, self.num_cells).zero_())
        d = Variable(typ(batch_size, self.num_cells).zero_())
        h = Variable(typ(batch_size, self.num_cells).zero_())
        a_max = Variable(typ(batch_size, self.num_cells).fill_(-1e38))
        return s, n, d, h, a_max

    # expects input of size (batch x 1 x num_features) - already split along time steps
    def forward(self, x_t, n_t, d_t, h_t, a_max_t):
        xh_join = torch.cat([x_t, h_t], 1).contiguous().view(x_t.size(0), -1)
        x_t = x_t.contiguous().view(x_t.size(0), -1)  # flatten

        u_t = self.u(x_t)
        g_t = self.g(xh_join)
        a_t = self.a(xh_join)

        z_t = u_t * Funct.tanh(g_t)

        if self.decay:
            decay = Funct.sigmoid(self.decay(xh_join))
            a_decay = a_max_t * torch.exp(-decay)
            a_newmax = torch.max(a_decay, a_t)
            exp_diff = torch.exp(a_max_t - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n_t * torch.exp(-decay) * exp_diff + z_t * exp_scaled  # update numerator
            d_t = d_t * torch.exp(-decay) * exp_diff + exp_scaled  # update denominator

        else:
            a_newmax = torch.max(a_max_t, a_t)
            exp_diff = torch.exp(a_max_t - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n_t * exp_diff + z_t * exp_scaled  # update numerator
            d_t = d_t * exp_diff + exp_scaled  # update denominator

        h_t = self.activation((n_t / d_t))  # update h
        a_max_t = a_newmax  # update a_max

        return n_t, d_t, h_t, a_max_t


class RWA(nn.Module):
    def __init__(self,
                 num_features,
                 num_cells,
                 num_classes,
                 decay=True,
                 init=1.0,
                 activation=Funct.tanh,
                 fwd_type="stepwise"):

        super(RWA, self).__init__()

        assert fwd_type == "stepwise" or fwd_type == "cumulative"

        if fwd_type == "stepwise":
            self.fwd_fn = self._fwd_stepwise
        elif fwd_type == "cumulative":
            self.fwd_fn = self._fwd_cumulative

        self.num_classes = num_classes
        self.activation = activation
        self.num_cells = num_cells

        self.cell = RWACell(num_features, num_cells, decay, init, activation)

        o_init_factor = np.sqrt((6.0 * init) / (num_cells + num_classes))

        self.o = nn.Linear(self.num_cells, self.num_classes)
        self.o.weight.data.uniform_(-o_init_factor, o_init_factor)
        self.o.bias.data.zero_()

    def cuda(self, device_id=None):
        self.cell._gpu = True
        super(RWA, self).cuda(device_id)

    def init_hidden(self, batch_size):
        return self.cell.init_hidden(batch_size)

    def _fwd_stepwise(self, x, n, d, h, a_max):
        outs = []

        h_t = h
        a_max_t = a_max
        n_t = n
        d_t = d
        for x_t in torch.unbind(x, 1):
            n_t, d_t, h_t, a_max_t = self.cell(x_t, n_t, d_t, h_t, a_max_t)

            o_t = self.o(h_t)
            outs.append(o_t)

        outs = torch.stack(outs, dim=1)
        return outs, n_t, d_t, h_t, a_max_t

    def _fwd_cumulative(self, x, n, d, h, a_max):

        h_t = h
        a_max_t = a_max
        n_t = n
        d_t = d
        for x_t in torch.unbind(x, 1):
            n_t, d_t, h_t, a_max_t = self.cell(x_t, n_t, d_t, h_t, a_max_t)

        outs = self.o(h_t)
        return outs, n_t, d_t, h_t, a_max_t

    def forward(self, x, hidden):  # x has shape (batch x steps x num_features)

        s = hidden[0]
        n = hidden[1]
        d = hidden[2]
        h = hidden[3]
        a_max = hidden[4]

        h_t = h + self.activation(s)

        outs, n_t, d_t, h_t, a_newmax = self.fwd_fn(x, n, d, h_t, a_max)

        return outs, (s, n_t, d_t, h_t, a_newmax)

    @staticmethod
    def detach_hidden(hidden):
        n = Variable(hidden[1].data)
        d = Variable(hidden[2].data)
        h = Variable(hidden[3].data)
        a_max = Variable(hidden[4].data)
        return (hidden[0], n, d, h, a_max)


class CGRURWACell(nn.Module):
    def __init__(self,
                 num_features,
                 time_steps,
                 num_filters,
                 num_classes):

        super(CGRURWACell, self).__init__()

        self.num_features = num_features
        self.time_steps = time_steps
        self.num_filters = num_filters

        self._gpu = False

        # Neural GPU that can take input at every time step and output at every time step
        # Use conv groups here, 1x1 convolution to get time info from sequence, and then run grouped conv
        # over the result to extract other info from that (see Xception https://arxiv.org/pdf/1610.02357.pdf)
        self.i = nn.Sequential(
            nn.Conv2d(time_steps + num_filters, num_filters, (1, 1))
        )

        self.g = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (1, 3), padding=(0, 1), groups=int(num_filters / 1)),
            nn.BatchNorm2d(num_filters)
        )

        self.a = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (1, 5), padding=(0, 2), groups=int(num_filters / 1)),
            nn.BatchNorm2d(num_filters)
        )

        self.ga = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (1, 1), groups=num_classes)  # this groups parameter (c/sh)ould change
        )

    def cuda(self, device_id=None):
        self._gpu = True
        super(CGRURWACell, self).cuda(device_id)

    def init_hidden(self, batch_size):
        typ = torch.FloatTensor
        if self._gpu:
            typ = torch.cuda.FloatTensor

        h = Variable(typ(batch_size, self.num_filters, 1, self.num_features).uniform_())

        return h

    def forward(self, x, h):
        xh_join = torch.cat([x, h], 1)

        i_t = self.i(xh_join)

        g_t = Funct.sigmoid(self.g(i_t))  # no sigmoid works well here, but I think it's better to leave this one
        a_t = self.a(i_t)

        h = g_t * i_t + (1.0 - g_t) * Funct.tanh(self.ga(i_t * a_t))

        return h


class CGRURWA(nn.Module):  # need to clean up this code
    def __init__(self,
                 num_features,
                 time_steps,
                 num_filters,
                 num_classes,
                 output_type='cumulative'):
        super(CGRURWA, self).__init__()

        self.num_features = num_features
        self.time_steps = time_steps
        self.num_filters = num_filters

        self.cell = CGRURWACell(num_features, time_steps, num_filters, num_classes)

        self.o = nn.Sequential(
            nn.Conv2d(num_filters, num_classes * (1 if output_type == 'cumulative' else time_steps), (1, 1)),
            nn.MaxPool2d((1, num_features), stride=(1, 1))
        )

    def cuda(self, device_id=None):
        self.cell._gpu = True
        super(CGRURWA, self).cuda(device_id)

    def init_hidden(self, batch_size):
        h = self.cell.init_hidden(batch_size)
        return h

    def forward(self, x, h):

        if x.dim() != 4:
            x = x.unsqueeze(2)

        if self.time_steps == x.size(1):
            h = self.cell(x, h)
        else:
            for i in range(int(x.size(1)/self.time_steps)):
                h = self.cell(x[:, i*self.time_steps:(i+1)*self.time_steps, :, :], h)

        outs = self.o(h)

        return outs, h

    @staticmethod
    def detach_hidden(h):
        h = Variable(h.data)
        return h


# Deprecated
class RWAGPU(nn.Module):
    def __init__(self,
                 num_features,
                 kernel_width,
                 num_filters,
                 num_classes,
                 init=1.0,
                 activation=Funct.tanh,
                 fwd_type="stepwise"):
        super(RWAGPU, self).__init__()

        assert fwd_type == "stepwise" or fwd_type == "cumulative"

        # if fwd_type == "stepwise":
        #     self.fwd_fn = self._fwd_stepwise
        # elif fwd_type == "cumulative":
        #     self.fwd_fn = self._fwd_cumulative

        self.num_features = num_features
        self.kernel_width = kernel_width
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.activation = activation
        self.init = init
        self.num_cells = self.num_features * self.num_filters

        self.x_resize = nn.Linear(self.num_features, self.num_cells, bias=False)

        self.g = nn.Sequential(
            nn.Conv2d(self.num_filters+1, self.num_filters, (1, kernel_width),
                           padding=(0, int(np.floor(kernel_width/2)))),
            nn.BatchNorm2d(self.num_filters)
        )
        self.u = nn.Sequential(
            nn.Conv2d(1, self.num_filters, (1, kernel_width),
                           padding=(0, int(np.floor(kernel_width/2)))),
            nn.BatchNorm2d(self.num_filters)
        )
        self.a = nn.Sequential(
            nn.Conv2d(self.num_filters+1, self.num_filters, (1, kernel_width),
                           padding=(0, int(np.floor(kernel_width/2)))),
            nn.BatchNorm2d(self.num_filters)
        )

        self.decay = nn.Sequential(
            nn.Conv2d(self.num_filters+1, self.num_filters, (1, kernel_width),
                           padding=(0, int(np.floor(kernel_width/2)))),
            nn.BatchNorm2d(self.num_filters)
        )

        o_init_factor = np.sqrt((6.0 * init) / (self.num_cells + self.num_classes))

        self.o = nn.Linear(self.num_cells*self.num_filters, self.num_classes)
        self.o.weight.data.uniform_(-o_init_factor, o_init_factor)
        self.o.bias.data.zero_()

    def init_sndha(self, batch_size):
        s = nn.Parameter(torch.FloatTensor(batch_size, self.num_filters, 1, self.num_cells).normal_(0.0, self.init))
        n = Variable(torch.zeros(batch_size, self.num_filters, 1, self.num_cells))
        d = Variable(torch.zeros(batch_size, self.num_filters, 1, self.num_cells))
        h = Variable(torch.zeros(batch_size, self.num_filters, 1, self.num_cells))
        a_max = Variable(torch.FloatTensor(batch_size, self.num_filters, 1, self.num_cells).fill_(-1e38))
        return s, n, d, h, a_max

    def forward(self, x, s, n, d, h, a_max):

        h = h + self.activation(s)

        # outs = []

        h_t = h
        a_max_t = a_max
        n_t = n
        d_t = d
        for x_t in torch.unbind(x, 1):
            x_t.contiguous()
            x_t = x_t.view(x_t.size(0), -1)
            x_t = self.x_resize(x_t)
            x_t = x_t.unsqueeze(1).unsqueeze(2)

            xh_join = torch.cat([x_t, h_t], 1)

            g_t = self.g(xh_join)
            u_t = self.u(x_t)
            a_t = self.a(xh_join)

            decay = Funct.sigmoid(self.decay(xh_join))

            z_t = u_t * Funct.tanh(g_t)

            a_decay = a_max_t * torch.exp(-decay)
            a_newmax = torch.max(a_decay, a_t)  # update a_max
            exp_diff = torch.exp(a_max_t - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n_t * torch.exp(-decay) * exp_diff + z_t * exp_scaled  # update numerator
            d_t = d_t * torch.exp(-decay) * exp_diff + exp_scaled  # update denominator

            h_t = self.activation((n_t / d_t))  # update h
            a_max_t = a_newmax  # update a_max

            # outs.append(self.o(h_t.view(h_t.size(0), -1)))

        # outs = torch.stack(outs, dim=1)  # need both forward types
        outs = self.o(h_t.view(h_t.size(0), -1))
        return outs, s, n_t, d_t, h_t, a_max_t


if __name__ == "__main__":
    rwagpu = RWAGPU(2, 1, 12, 1)
    s, n, d, h, a = rwagpu.init_sndha(1)
    input = Variable(torch.rand(1, 100, 1, 2))
    print(rwagpu(input, s, n, d, h, a))
