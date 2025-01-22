#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion3d import H36motion3D
import utils.model2 as nnmodel
# import utils.no_gru as nnmodel
import utils.data_utils as data_utils

# torch.backends.cudnn.enabled = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_3D_in{:d}_out{:d}_dct_n_{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=66)#源代码node_n=66

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        model_path_len = 'H:\whz\CODE\model/' + 'ckpt_' + script_name + '_last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                                split=0, dct_used=dct_n, sample_rate=sample_rate)

    acts = data_utils.define_actions('all')
    test_data = dict()
    # *******获取所有动作的平均误差

    all_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n, split=1,
                               sample_rate=sample_rate, dct_used=dct_n)
    all_dataloader = DataLoader(
            dataset=all_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=0,#opt.job
            pin_memory=True)

    # *******
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                   sample_rate=sample_rate, dct_used=dct_n)

        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=0,#opt.job
            pin_memory=True)
    val_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                              split=2, dct_used=dct_n, sample_rate=sample_rate)

    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    small = 100
    use = []
    num = [[] for i in range(len(acts))]
    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda,
                            dim_used=train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=dct_n)

        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])

        # *******获取所有动作的平均误差
        test_3d, limb_ls, even_point = tes(all_dataloader, model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                                                      dim_used=train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, test_3d)
        head = np.append(head, ['3d80', '3d160', '3d320', '3d400', '3d560', '3d1000'])
        # *******

        for type, act in enumerate(acts):
            test_3d, limb_ls, even_point = tes(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=dct_n)
            if v_3d < small:
                small = v_3d
                num[type] = even_point

            # ret_log = np.append(ret_log, test_l)
            ret_log = np.append(ret_log, test_3d)
            head = np.append(head, [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            limb = ['right_leg', 'left_leg', 'right_hand', 'left_hand', 'trunk']
            # for i, name in zip(range(len(limb_ls)), limb):
                # ret_log = np.append(ret_log, limb_ls[i])
                # head = np.append(head, [name + '3d80', name + '3d160', name + '3d320', name + '3d400'])
            if output_n > 10:
                head = np.append(head, [act + '3d560', act + '3d1000'])
                # for i, name in zip(range(len(limb_ls)), limb):
                #     ret_log = np.append(ret_log, limb_ls[i])
                #     head = np.append(head, [name + '3d80', name + '3d160', name + '3d320', name + '3d400', name + '3d560', name + '3d1000'])

        all = []
        if epoch == opt.epochs:
            pd.DataFrame(num).to_csv(r'H:\whz\CODE\model\all act.csv')
            for i in num:
               all = np.add(all, i)
            all = all / len(num)
            pd.DataFrame(all).to_csv('H:\whz\CODE\model\sample.csv')
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0)) 
        if epoch == 0:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], dct_n=15):
    t_l = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = Variable(inputs.cuda(non_blocking=True)).float()
            # all_seq = Variable(all_seq.cuda(async=True)).float()
            all_seq = all_seq.cuda(non_blocking=True).float()

        outputs = model(inputs)
        # outputs = inputs
        # calculate loss and backward
        loss = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        t_l.update(loss.cpu().data.numpy().reshape(1,-1)[0] * batch_size, batch_size)  #  t_l.update(loss.cpu().data.numpy()[0] * batch_size, batch_size)原代码
        # print('****************', loss, t_l.avg, t_l.val, t_l.sum, t_l.count)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg


def tes(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    t_l = utils.AccumLoss()
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))
    right_leg = np.zeros(len(eval_frame))
    left_leg = np.zeros(len(eval_frame))
    right_hand = np.zeros(len(eval_frame))
    left_hand = np.zeros(len(eval_frame))
    trunk = np.zeros(len(eval_frame))
    lleg_use = [1, 2, 3, 4, 5]
    rleg_use = [6, 7, 8, 9, 10]
    lhand_use = [25, 26, 27, 28, 29, 30, 31]
    rhand_use = [17, 18, 19, 20, 21, 22, 23]
    trunk_use = [0, 11, 12, 13, 14, 15, 16, 24]
    limb_ls = [right_leg, left_leg, right_hand, left_hand, trunk]
    limb_use = [lleg_use, rleg_use, lhand_use, rhand_use, trunk_use]
    even_point = []
    for i in range(32):
        chip = np.zeros(len(eval_frame))
        even_point.append(chip)

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        batch_size = inputs.shape[0]

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # all_seq = Variable(all_seq.cuda(async=True)).float()
            all_seq = all_seq.cuda(non_blocking=True).float()

        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)


        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.contiguous().view(-1, dct_n).transpose(0, 1)
        outputs_3d = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                   seq_len).transpose(1, 2)
        # outputs_3d = outputs.transpose(1, 2)
        # out = outputs_3d.cpu().data.numpy()
        pred_3d = all_seq.clone()
        dim_used = np.array(dim_used)

        # joints at same loc
        joint_to_ignore = np.array([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]) #[0, 1, 6, 11, 16, 20, 23, 24, 28, 31]
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        # joint_equal = np.array([13, 19, 22, 13, 27, 30]) 改
        joint_equal = np.array([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        pred_3d[:, :, dim_used] = outputs_3d
        # pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        #*************************计算各个点的误差**************************
        for i, chip in enumerate(even_point):
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                chip[k] += torch.mean(torch.norm(
                    targ_p3d[:, j, i, :].contiguous().view(-1, 3) - pred_p3d[:, j, i, :].contiguous().view(-1, 3),
                    2,1)).cpu().data.numpy().reshape(1, -1)[0] * n  # 改
        # ************************计算分肢的误差**********************
        # for limb, use in zip(limb_ls, limb_use):
        #     for k in np.arange(0, len(eval_frame)):
        #         j = eval_frame[k]
        #         limb[k] += torch.mean(torch.norm(
        #             targ_p3d[:, j, use, :].contiguous().view(-1, 3) - pred_p3d[:, j, use, :].contiguous().view(-1, 3), 2,
        #             1)).cpu().data.numpy().reshape(1,-1)[0] *n #改

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy().reshape(1,-1)[0] *n #改
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    # for i in range(len(limb_ls)):
    #     limb_ls[i] = limb_ls[i]/N
    for chip in even_point:
        for i in range(len(chip)):
            chip[i] = chip[i]/N
    return t_3d/N, limb_ls, even_point


def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # all_seq = Variable(all_seq.cuda(async=True)).float()
            all_seq = all_seq.cuda(non_blocking=True).float()

        outputs = model(inputs)

        n, _, _ = all_seq.data.shape

        m_err = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
        trunk_index = []

        # update the training loss
        t_3d.update(m_err.cpu().data.numpy().reshape(1,-1)[0] * n, n)#改

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
