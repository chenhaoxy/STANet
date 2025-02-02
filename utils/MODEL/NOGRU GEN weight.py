#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from ..data_utils import get_dct_matrix

from ..tools import Graph
import torch.nn.functional as F


class GR_AdjMatrix(nn.Module):
    def __init__(self, input_size, hidden_size, node_num):
        super(GR_AdjMatrix, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.weight_xz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))

        self.weight_xr = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))

        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h, a):
        x = F.linear(x, a)
        # 更新门
        z = torch.sigmoid(F.linear(x, self.weight_xz, self.bias_z) + F.linear(h, self.weight_hz))
        # 重置门
        r = torch.sigmoid(F.linear(x, self.weight_xr, self.bias_r) + F.linear(h, self.weight_hr))
        # 候选隐藏状态
        h_tilde = torch.tanh(F.linear(x, self.weight_xh, self.bias_h) + F.linear(r * h, self.weight_hh))
        # 最终隐藏状态
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, rel=False, bias=True, node_n=48): #
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))

        # self.do = nn.Dropout(p_dropout)
        self.bone = 0
        # self.larm_rleg = 0
        # self.larm_lleg = 0
        # self.rarm_rleg = 0
        # self.rarm_lleg = 0
        # self.trunk_lleg = 0
        # self.trunk_rleg = 0
        # self.trunk_larm = 0
        # self.trunk_rarm = 0
        # self.lrarm = 0
        # self.lrleg = 0
        self.down_lamb = 0
        # self.CeN = CeN(node_n, in_features)  # 实例化！！！！！！！！！！！！

        # self.bone = Parameter(torch.FloatTensor(1))
        self.larm_rleg = Parameter(torch.FloatTensor(1))
        self.larm_lleg = Parameter(torch.FloatTensor(1))
        self.rarm_rleg = Parameter(torch.FloatTensor(1))
        self.rarm_lleg = Parameter(torch.FloatTensor(1))
        self.trunk_lleg = Parameter(torch.FloatTensor(1))
        self.trunk_rleg = Parameter(torch.FloatTensor(1))
        self.trunk_larm = Parameter(torch.FloatTensor(1))
        self.trunk_rarm = Parameter(torch.FloatTensor(1))
        self.lrarm = Parameter(torch.FloatTensor(1))
        self.lrleg = Parameter(torch.FloatTensor(1))
        # self.down_lamb = Parameter(torch.FloatTensor(1))
        # self.w1 = Parameter(torch.FloatTensor(1))
        # self.w2 = Parameter(torch.FloatTensor(1))
        # self.w3 = Parameter(torch.FloatTensor(1))
        # self.w4 = Parameter(torch.FloatTensor(1))


        self.pre = Graph(node_n).get_edge()
        self.edge = torch.zeros((node_n, node_n)).cuda()

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)

        # self.bone.data.uniform_(-stdv, stdv)
        self.larm_rleg.data.uniform_(-stdv, stdv)
        self.larm_lleg.data.uniform_(-stdv, stdv)
        self.rarm_rleg.data.uniform_(-stdv, stdv)
        self.rarm_lleg.data.uniform_(-stdv, stdv)
        self.trunk_lleg.data.uniform_(-stdv, stdv)
        self.trunk_rleg.data.uniform_(-stdv, stdv)
        self.trunk_larm.data.uniform_(-stdv, stdv)
        self.trunk_rarm.data.uniform_(-stdv, stdv)
        self.lrarm.data.uniform_(-stdv, stdv)
        self.lrleg.data.uniform_(-stdv, stdv)
        # self.down_lamb.data.uniform_(-stdv, stdv)
        # self.w1.data.uniform_(-stdv, stdv)
        # self.w2.data.uniform_(-stdv, stdv)
        # self.w3.data.uniform_(-stdv, stdv)
        # self.w4.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, h):  # shape[b, node ,seq_len]

        if self.rel:  # h.shape = [layers, batch, hidden_size]

            # i = int(h.shape[0]/2)
            # h = h[i]
            self.edge = self.bone * self.pre[0] + h[0] * self.larm_rleg * self.pre[1] + h[1] * self.larm_lleg *\
                        self.pre[2] + h[2] * self.rarm_rleg * self.pre[3]+ h[3] * self.rarm_lleg * self.pre[4] +\
                        h[4] * self.trunk_lleg * self.pre[5] + h[5] * self.trunk_rleg * self.pre[6] + \
                        h[6] * self.trunk_larm * self.pre[7] + h[7] * self.trunk_rarm * self.pre[8] + \
                        h[8] * self.lrarm * self.pre[9] + h[9] * self.lrleg * self.pre[10] + self.down_lamb * self.pre[11]
        else:
            self.edge = self.bone * self.pre[0] + self.larm_rleg * self.pre[1] + self.larm_lleg * self.pre[2] + \
                        self.rarm_rleg * self.pre[3] + self.rarm_lleg * self.pre[4] + self.trunk_lleg * self.pre[5] + \
                        self.trunk_rleg * self.pre[6] + self.trunk_larm * self.pre[7] + self.trunk_rarm * self.pre[8] + \
                        self.lrarm * self.pre[9] + self.lrleg * self.pre[10] + self.down_lamb * self.pre[11]
        # self.edge = self.w1 * self.pre[0] + self.w2 * self.pre[1] + self.w3 * self.pre[2] + self.w4 * self.pre[3]

        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att + self.edge, support) #
        if self.bias is not None:
            return output + self.bias
        else:
            return output,

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, rel=False) #
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gru2 = torch.nn.GRU(input_size=in_features, hidden_size=in_features, num_layers=2, batch_first=True)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, rel=False) # *****修改2024.4.5*****
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, h): #

        # y, _ = self.gru1(x)

        y = self.gc1(x, h) #
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.gru2(y)

        y = self.gc2(y, h) #
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        dct,idct = get_dct_matrix(20)
        self.dct = torch.tensor(dct).cuda().float()
        self.idct = torch.tensor(idct).cuda().float()
        self.input_feature = input_feature
        self.A = Graph(node_n).get_edge()
        self.num_stage = num_stage

        # self.gru = nn.GRU(66, 256, num_layers=3, batch_first=True)
        # self.gru2 = nn.GRU(256, 66, num_layers=3, batch_first=True)
        # # self.gru1 = GR_AdjMatrix(66, 66, 66)
        # self.gru1 = nn.GRUCell(66, 66)
        # self.gen_weight = nn.Linear(10, 10)

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n, rel=False) #
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n, rel=False) #

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):

        weight = []
        # b = int(x.shape[0]/2)
        # seq_len = x.shape[2]
        # x1 = x.transpose(2, 1)
        # gru_x, h = self.gru(x1)
        # gru_x, h = self.gru2(gru_x)
        # gru_x = gru_x.transpose(1, 0)
        # h1 = h[2]
        # for n in range(10):
        #     h2 = torch.FloatTensor()
        #     for i in range(seq_len):
        #         h1 = self.gru1(F.linear(gru_x[i], self.A[n+1]), h1)
        #         if i == 9:
        #             h2 = h1
        #     weight.append(torch.sum(torch.abs(h1[b]-h2[b])))
        # weight = self.gen_weight(torch.tensor(weight).to('cuda'))
        y = self.gc1(x, weight) #
        b, n, f = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):  # modulelist和sequential的使用区别
            y = self.gcbs[i](y, weight) #

        y = self.gc7(y, weight) #
        y = y + x

        return y

