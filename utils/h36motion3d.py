from torch.utils.data import Dataset
import numpy as np
from h5py import File
# import scipy.io as sio
import utils.data_utils as data_utils
from matplotlib import pyplot as plt


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, dct_used=35, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = input_n + output_n

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])#subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        if split == 3:
            subjs = [-1]
            all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                     dct_used)
        else:
            subjs = subs[split]
            all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                     dct_used)
        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        # output = all_seqs
        all_seqs = all_seqs.reshape(-1, dct_used)
        all_seqs = all_seqs.transpose()
         # **************将坐标转化为DCT系数*******************
        dct_m_in, _ = data_utils.get_dct_matrix(dct_used)
        dct_m_out, _ = data_utils.get_dct_matrix(dct_used)
        pad_idx = np.repeat([input_n - 1], dct_used-input_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx) #目的是生成最终的索引数组，也就是将input_n的最后一个元素复制output_n次
        # all_seqs = all_seqs[:, :, i_idx]
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # input_dct_seq = input_dct_seq.transpose(0, 2, 1)  # 针对simlps的数据处理
        # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)

        output_dct_seq = np.matmul(dct_m_out[0:dct_used, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # output_dct_seq = output_dct_seq.transpose(0, 2, 1)  # 针对simlps的数据处理
        # output_dct_seq = output_dct_seq.reshape(-1, len(dim_used) * dct_used)
        input_dct_seq = input_dct_seq
        output_dct_seq = output_dct_seq

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
