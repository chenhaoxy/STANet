import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Globel_state(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Globel_state, self).__init__()
        self.hidden_size = hidden_size

        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = Parameter(torch.Tensor(hidden_size))

        self.weight_hc = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gc = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_g = Parameter(torch.Tensor(hidden_size))

        self.weight_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_go = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_o = Parameter(torch.Tensor(hidden_size))

        self.relu = nn.Sigmoid()
        self.act = nn.Tanh()
        # self.g_ = torch.zeros([128, 66]).to('cuda')
        # self.f_j = torch.zeros([128, 66]).to('cuda')

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, h, c, g, c_g): # h是当前cell的所有隐藏状态 g是上一个cell的全局状态 c是当前cell的所有中间状态
        g_ = torch.zeros(h[0].size()).to('cuda')
        f_j = torch.zeros(h[0].size()).to('cuda')
        for h_ in h:
            g_ += h_/len(h)

        for h_, c_ in zip(h, c):
            f_j += self.relu(F.linear(h_, self.weight_hh, self.bias_h) + F.linear(g, self.weight_gh)) * c_

        f_g = self.relu(F.linear(g_, self.weight_hc, self.bias_g) + F.linear(g, self.weight_gc))

        o_g = self.relu(F.linear(g_, self.weight_ho, self.bias_o) + F.linear(g, self.weight_go))

        c_g = f_j + f_g*c_g

        g_final = o_g*self.act(c_g)
        return g_final, c_g



class GR_A(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GR_A, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.weight_xz = Parameter(torch.Tensor(hidden_size, input_size))  #*h
        self.weight_hz = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gz = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_z = Parameter(torch.Tensor(hidden_size))

        self.weight_xi = Parameter(torch.Tensor(hidden_size, input_size))  # *cg
        self.weight_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_i = Parameter(torch.Tensor(hidden_size))

        self.weight_xr = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gr = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = Parameter(torch.Tensor(hidden_size))

        self.weight_xh = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_gh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = Parameter(torch.Tensor(hidden_size))

        self.weight_xo = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_go = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_o = Parameter(torch.Tensor(hidden_size))

        self.relu = nn.Sigmoid()
        self.act = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h, a, g): # x看做上一个cell的对应位置的隐藏状态，h是上一个时间片处理后输出的隐藏状态，a是矩阵，g是上一个cell的全局状态
        x = F.linear(x, a)
        # print("1:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
        # 更新门
        z = self.relu(F.linear(x, self.weight_xz, self.bias_z) + F.linear(h, self.weight_hz) +
                          F.linear(g, self.weight_gz))
        # 全局门
        i = self.relu(F.linear(x, self.weight_xi, self.bias_z) + F.linear(h, self.weight_hi) +
                          F.linear(g, self.weight_gi))
        # 重置门
        r = self.relu(F.linear(x, self.weight_xr, self.bias_r) + F.linear(h, self.weight_hr) +
                          F.linear(g, self.weight_gr))
        # print("2:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
        # 候选隐藏状态
        h_tilde = self.act(F.linear(x, self.weight_xh, self.bias_h) + F.linear(r * h, self.weight_hh) +
                             F.linear(g, self.weight_gh))
        # 中间隐藏状态
        c = (1 - z) * x + z * h_tilde + i*h
        # 总控制门
        o = self.relu(F.linear(x, self.weight_xo, self.bias_o) + F.linear(h, self.weight_ho) +
                          F.linear(g, self.weight_go))
        # print("4:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
        h_next = o * self.act(c)

        return h_next, c

class GR_Weight(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GR_Weight, self).__init__()
        self.hidden_size = hidden_size

        # self.GR_A1 = nn.ModuleList([GR_A(input_size, hidden_size),
        #                            GR_A(hidden_size, hidden_size),
        #                            GR_A(hidden_size, hidden_size)])
        # self.G_state1 = nn.ModuleList([Globel_state(input_size, hidden_size),
        #                               Globel_state(input_size, hidden_size),
        #                               Globel_state(input_size, hidden_size)])
        self.GR_A1 = GR_A(input_size, hidden_size)
        self.G_state1 = Globel_state(input_size, hidden_size)


        # self.h = torch.zeros(hidden_size).to('cuda')#是这些的问题
        # self.g = torch.ones(hidden_size).to('cuda')
        # self.c_g = torch.ones(hidden_size).to('cuda')
        # self.g_final = torch.zeros(self.hidden_size).to('cuda')
        # self.c_g = torch.zeros(self.hidden_size).to('cuda')

    def forward(self, x, a):# 不使用列表保存结果就不会增加内存，不使用循环也不会增加
        h1 = torch.zeros(self.hidden_size).to('cuda')
        g1 = torch.ones(self.hidden_size).to('cuda')
        c_g = torch.ones(self.hidden_size).to('cuda')
        h = []
        c = []
        for i in x:
            h_next, c_ = self.GR_A1(i, h1, a, g1)
            h1 = h_next
            # h2 = h_next.cpu().detach().numpy()
            # plt.matshow(h2, cmap=plt.get_cmap('autumn'), alpha=0.5)  # , alpha=0.3
            # plt.show()
            h.append(h1)
            c.append(c_)
        g_final, c_g = self.G_state1(h, c, g1, c_g)
        # *******
        # for GR, G_s in zip(self.GR_A1, self.G_state1):
        #     h = []
        #     c = []
        #     for i in x:
        #         h_next, c_ = GR(i, self.h, a, self.g)
        #         self.h = h_next
        #         h.append(self.h)
        #         c.append(c_)
        #     self.g_final, self.c_g = G_s(h, c, self.g, self.c_g)
        # *******
        # for GR, G_s in zip(self.GR_A1, self.G_state1):
        #     h = []
        #     c = []
        #     for i in x:
        #         h_next, c_ = GR(i, h1, a, g1)
        #         h1 = h_next
        #         h.append(h1)
        #         c.append(c_)
        #     g_final, c_g = G_s(h, c, g1, c_g)

        return g_final, c_g



