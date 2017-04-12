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


class RWA(nn.Module):
    def __init__(self,
                 num_features,
                 num_cells,
                 num_classes,
                 activation=Funct.tanh,
                 fwd_type="stepwise"):

        super(RWA, self).__init__()

        assert fwd_type == "stepwise" or fwd_type == "cumulative"

        if fwd_type == "stepwise":
            self.fwd_fn = self._fwd_stepwise
        elif fwd_type == "cumulative":
            self.fwd_fn = self._fwd_cumulative

        self.num_features = num_features
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.activation = activation

        init_factor = np.sqrt(6.0 / (num_features + 2.0 * num_cells))

        self.g = nn.Linear(self.num_features + self.num_cells, self.num_cells)
        self.g.weight.data.uniform_(-init_factor, init_factor)
        self.g.bias.data.zero_()

        self.u = nn.Linear(self.num_features, self.num_cells)
        self.u.weight.data.uniform_(-init_factor, init_factor)
        self.u.bias.data.zero_()

        self.a = nn.Linear(self.num_features + self.num_cells, self.num_cells, bias=False)
        self.a.weight.data.uniform_(-init_factor, init_factor)

        self.o = nn.Linear(self.num_cells, self.num_classes)
        self.o.weight.data.uniform_(-init_factor, init_factor)
        self.o.bias.data.zero_()

    def init_sndha(self, batch_size):
        s = Variable(torch.FloatTensor(self.num_cells).normal_(0.0, 1.0), requires_grad=True)
        n = Variable(torch.zeros(batch_size, self.num_cells))
        d = Variable(torch.zeros(batch_size, self.num_cells))
        h = Variable(torch.zeros(batch_size, self.num_cells))
        a_max = Variable(torch.ones(batch_size, self.num_cells) * -1e38)
        # start with very negative number
        return s, n, d, h, a_max

    def _fwd_stepwise(self, x, n, d, h, a_max):
        outs = []

        for x_t in torch.unbind(x, 1):
            xh_join = torch.cat([x_t, h], 1)

            x_t = x_t.contiguous()
            x_t = x_t.view(x_t.size(0), -1)

            xh_join = xh_join.contiguous()
            xh_join = xh_join.view(xh_join.size(0), -1)

            u_t = self.u(x_t)
            g_t = self.g(xh_join)
            a_t = self.a(xh_join)

            z_t = u_t * Funct.tanh(g_t)

            a_newmax = torch.max(a_max, a_t)
            exp_diff = torch.exp(a_max - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n * exp_diff + z_t * exp_scaled
            d_t = d * exp_diff + exp_scaled

            h = self.activation(n_t / d_t)

            outs.append(self.o(h))

        outs = torch.stack(outs, dim=1)
        return outs, n_t, d_t, h, a_newmax

    def _fwd_cumulative(self, x, n, d, h, a_max):
        for x_t in torch.unbind(x, 1):
            xh_join = torch.cat([x_t, h], 1)

            x_t = x_t.contiguous()
            x_t = x_t.view(x_t.size(0), -1)

            xh_join = xh_join.contiguous()
            xh_join = xh_join.view(xh_join.size(0), -1)

            u_t = self.u(x_t)
            g_t = self.g(xh_join)
            a_t = self.a(xh_join)

            z_t = u_t * Funct.tanh(g_t)

            a_newmax = torch.max(a_max, a_t)
            exp_diff = torch.exp(a_max - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n * exp_diff + z_t * exp_scaled
            d_t = d * exp_diff + exp_scaled

            h = self.activation(n_t / d_t)

        outs = self.o(h)
        return outs, n_t, d_t, h, a_newmax

    def forward(self, x, s, n, d, h, a_max):  # x has shape (batch x steps x num_features)

        h = h + self.activation(s.repeat(x.size(0), 1))

        outs, n_t, d_t, h, a_newmax = self.fwd_fn(x, n, d, h, a_max)

        return outs, s, n_t, d_t, h, a_newmax



