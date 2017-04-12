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
                 init=1.0,
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
        self.init = init

        ga_init_factor = np.sqrt((6.0 * init) / (num_features + 2.0 * num_cells))
        u_init_factor = np.sqrt((6.0 * init) / (num_features + num_cells))
        o_init_factor = np.sqrt((6.0 * init) / (num_cells + num_classes))

        self.g = nn.Linear(self.num_features + self.num_cells, self.num_cells)
        self.g.weight.data.uniform_(-ga_init_factor, ga_init_factor)
        self.g.bias.data.zero_()
        # self.g_drop = nn.Dropout(p=0.2)

        self.u = nn.Linear(self.num_features, self.num_cells)
        self.u.weight.data.uniform_(-u_init_factor, u_init_factor)
        self.u.bias.data.zero_()
        # self.u_drop = nn.Dropout(p=0.2)

        self.a = nn.Linear(self.num_features + self.num_cells, self.num_cells, bias=False)
        self.a.weight.data.uniform_(-ga_init_factor, ga_init_factor)
        # self.a_drop = nn.Dropout(p=0.2)

        self.o = nn.Linear(self.num_cells, self.num_classes)
        self.o.weight.data.uniform_(-o_init_factor, o_init_factor)
        self.o.bias.data.zero_()
        # self.o_drop = nn.Dropout(p=0.05)

    def init_sndha(self, batch_size):
        s = nn.Parameter(torch.FloatTensor(self.num_cells).normal_(0.0, self.init), requires_grad=True)
        n = Variable(torch.zeros(batch_size, self.num_cells))
        d = Variable(torch.zeros(batch_size, self.num_cells))
        h = Variable(torch.zeros(batch_size, self.num_cells))
        a_max = Variable(torch.FloatTensor(batch_size, self.num_cells).fill_(-1e38))
        return s, n, d, h, a_max

    def _fwd_stepwise(self, x, n, d, h, a_max):
        outs = []

        h_t = h
        a_max_t = a_max
        n_t = n
        d_t = d
        for x_t in torch.unbind(x, 1):  # Unbind the tensor along the time/steps dimension
            xh_join = torch.cat([x_t, h_t], 1)  # concat the time step input with the time step h

            x_t = x_t.contiguous()
            x_t = x_t.view(x_t.size(0), -1)  # flatten time step input

            xh_join = xh_join.contiguous()
            xh_join = xh_join.view(xh_join.size(0), -1)  # flatten time step h

            # Gates, u, g, a
            # u_t = self.u_drop(x_t)
            u_t = self.u(x_t)

            # g_t = self.g_drop(xh_join)
            g_t = self.g(xh_join)

            # a_t = self.a_drop(xh_join)
            a_t = self.a(xh_join)

            z_t = u_t * Funct.tanh(g_t)  # pointwise multiply

            a_newmax = torch.max(a_max_t, a_t)  # update a_max
            exp_diff = torch.exp(a_max_t - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n_t * exp_diff + z_t * exp_scaled  # update numerator
            d_t = d_t * exp_diff + exp_scaled  # update denominator

            h_t = self.activation((n_t / d_t))  # update h
            a_max_t = a_newmax  # update a_max

            # o_t = self.o_drop(h_t)
            o_t = self.o(h_t)
            outs.append(o_t)

        outs = torch.stack(outs, dim=1)
        return outs, n_t, d_t, h_t, a_max_t

    def _fwd_cumulative(self, x, n, d, h, a_max):

        h_t = h
        a_max_t = a_max
        n_t = n
        d_t = d
        for x_t in torch.unbind(x, 1):  # make list of (batch x features) that is (steps) long
            xh_join = torch.cat([x_t, h_t], 1)  # concat the time step input with the time step h

            x_t = x_t.contiguous()
            x_t = x_t.view(x_t.size(0), -1)  # flatten time step input

            xh_join = xh_join.contiguous()
            xh_join = xh_join.view(xh_join.size(0), -1)  # flatten time step h

            # Gates, u, g, a
            # u_t = self.u_drop(x_t)
            u_t = self.u(x_t)

            # g_t = self.g_drop(xh_join)
            g_t = self.g(xh_join)

            # a_t = self.a_drop(xh_join)
            a_t = self.a(xh_join)

            z_t = u_t * Funct.tanh(g_t)  # pointwise multiply

            a_newmax = torch.max(a_max_t, a_t)  # update a_max
            exp_diff = torch.exp(a_max_t - a_newmax)
            exp_scaled = torch.exp(a_t - a_newmax)

            n_t = n_t * exp_diff + z_t * exp_scaled  # update numerator
            d_t = d_t * exp_diff + exp_scaled  # update denominator

            h_t = self.activation((n_t / d_t))  # update h
            a_max_t = a_newmax  # update a_max

        # outs = self.o_drop(h_t)
        outs = self.o(h_t)
        return outs, n_t, d_t, h_t, a_max_t

    def forward(self, x, s, n, d, h, a_max):  # x has shape (batch x steps x num_features)

        h_t = h + self.activation(s.repeat(x.size(0), 1))

        outs, n_t, d_t, h_t, a_newmax = self.fwd_fn(x, n, d, h_t, a_max)

        return outs, s, n_t, d_t, h_t, a_newmax



