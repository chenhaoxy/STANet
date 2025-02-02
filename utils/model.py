#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from utils.data_utils import get_dct_matrix
from .MODEL.GRW import GR_Weight

from .tools import Graph
from .tools import Enhance
import torch.nn.functional as F




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
        self.rel = rel


        self.pre = Graph(node_n, '3dpw').get_edge()
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

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, h):  # shape[b, node ,seq_len]

        if self.rel:
            self.edge = h[0] * self.larm_rleg * self.pre[0] + h[1] * self.larm_lleg *self.pre[1] + \
                        h[2] * self.rarm_rleg * self.pre[2]+ h[3] * self.rarm_lleg * self.pre[3] +\
                        h[4] * self.trunk_lleg * self.pre[4] + h[5] * self.trunk_rleg * self.pre[5] + \
                        h[6] * self.trunk_larm * self.pre[6] + h[7] * self.trunk_rarm * self.pre[7] + \
                        h[8] * self.lrarm * self.pre[8] + h[9] * self.lrleg * self.pre[9]
        #
        # else:
        #     self.edge = self.larm_rleg * self.pre[0] + self.larm_lleg * self.pre[1] + \
        #                 self.rarm_rleg * self.pre[2] + self.rarm_lleg * self.pre[3] + \
        #                 self.trunk_lleg * self.pre[4] + self.trunk_rleg * self.pre[5] + \
        #                 self.trunk_larm * self.pre[6] + self.trunk_rarm * self.pre[7] + \
        #                 self.lrarm * self.pre[8] + self.lrleg * self.pre[9]


        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att + self.edge, support)#

        if self.bias is not None:
            return output + self.bias
        else:
            return output

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

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, rel=True) #
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gru2 = torch.nn.GRU(input_size=in_features, hidden_size=in_features, num_layers=2, batch_first=True)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, rel=True)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, h): #

        # y, _ = self.gru2(x)

        y = self.gc1(x, h)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y, _ = self.gru2(y)

        y = self.gc2(y, h)
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
        self.input_feature = input_feature
        self.A = Graph(node_n).get_edge()
        self.num_stage = num_stage

        self.GR_Weight = GR_Weight(node_n, node_n)
        self.gen_weight = nn.Linear(10, 10)

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n, rel=True) #
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n, rel=True) #

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):
        weight = []
        b = int(x.shape[0]/2)
        seq_len = x.shape[2]
        x1 = x.transpose(2, 1)
        x1 = x1.transpose(1, 0)
        for n in range(10):
            g, _ = self.GR_Weight(x1, self.A[n])
            w = torch.mean(torch.abs(g-x1[9]))
            weight.append(w)
        weight = self.gen_weight(torch.tensor(weight).to('cuda'))
        y = self.gc1(x, weight) #
        b, n, f = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):  # modulelist和sequential的使用区别
            y = self.gcbs[i](y, weight) #

        y = self.gc7(y, weight) #
        y = y + x

        return y#, weight #返回两个tensor但是只是用一个变量接着就会自动保存成一个元组

