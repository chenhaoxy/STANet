import math

import torch
import torch.nn as nn
from utils.MODEL.GRW import Globel_state
from utils.MODEL.GRW import GR_A
import torch.nn.functional as F


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


        self.h = torch.zeros(hidden_size).to('cuda')
        self.g = torch.ones(hidden_size).to('cuda')
        self.c_g = torch.ones(hidden_size).to('cuda')
        self.g_final = torch.zeros(self.hidden_size).to('cuda')
        # self.c_g = torch.zeros(self.hidden_size).to('cuda')

    def forward(self, x, a):# 不使用列表保存结果就不会增加内存，不使用循环也不会增加
        h = []
        c = []
        for i in x:
            h_next, c_ = self.GR_A1(i, self.h, a, self.g)
            self.h = h_next
        h.append(self.h)
        c.append(c_)
        self.g_final, self.c_g = self.G_state1(h, c, self.g, self.c_g)

        return self.g_final, self.c_g