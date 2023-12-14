#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
based on optuna AutoML framework
include checkpoint and early stopping
'''
import json
import argparse
import os
import logging
import sys
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter\

from util.read_data import SegmentationDataset
from model.Generator import Generator 
from model.Discriminator import Discriminator

import monai
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

import contextual_loss as cl
import warnings
warnings.filterwarnings('ignore') # avoid tensor error

torch.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(trial):
    ''' Objective function for AutoML, trial is the hyperparameter needed to be tuned '''

    # other hyperparameters that are not tuned
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/imgs', help='input RGB or Gray image path')
    parser.add_argument('--mask_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/masks', help='input mask path')
    parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--epoch", type=int, default=30, help="maximal epoch in training for every trial")
    # parser.add_argument("--val_batch", type=int, default=200, help="Every val_batch, do validation")
    # parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")
    args = parser.parse_args()

    # hyperparameters that are tuned
    batch_size = trial.suggest_int("batch_size", 4, 16,step=4)

    lr_G = trial.suggest_float("lr_G", 5e-4, 1e-2, log=True)  # 注意所有参数在trial中一旦确定无法更改-覆盖!!!
    lr_D = trial.suggest_float("lr_D", 1e-4, 1e-3, log=True) 

    adv_ratio = trial.suggest_float("adv_ratio", 0.1, 0.5, step=0.1)
    seg_ratio = trial.suggest_float("seg_ratio", 0.5, 1.0, step=0.1)
    con_ratio = trial.suggest_float("con_ratio", 0.1, 0.5, step=0.1)

    # parepare data
    dataset = SegmentationDataset(args.image_dir, args.mask_dir) 
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)]) # manual_seed fixed
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # prepare model
    generator = Generator().to(device)   # input channel must be 1
    discriminator = Discriminator().to(device) 

    # optimizer and loss
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(args.b1, args.b2))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(args.b1, args.b2))

    loss_adv = torch.nn.BCELoss().to(device) # GAN adverserial loss
    metric_val = monai.metrics.DiceHelper(sigmoid=True) 
    loss_seg =  monai.losses.FocalLoss(alpha=0.75, gamma=2.0).to(device)

    # other components
    Context_crit = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4',band_width=0.3).to(device) 
    tf = Compose([Activations(sigmoid=True)])

    # training
    batch_num = 0
    metric = 0
    
    for epoch in range(args.epoch): # epoch会始终保持0,因为optuna内核

        len_minibatch = dataloader_train.__len__()
        pbar = tqdm(total=len_minibatch, desc='Processing for this epoch in current trial')
        
        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            
            pbar.update(1)
            batch_num += 1 

            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator 1为真   
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator 0为假
            
            mask = mask.to(device).float()
            img = img.to(device) 

            generator.train()  # recover to train mode(because of eval in validation)
            discriminator.train()  # recover to train mode
            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()
            g_output = generator(img) 
            loss_seg_ = loss_seg(input=g_output, target=mask) # focal loss自带sigmoid
            g_output_norm = tf(g_output) # ([8, 1, 256, 256])
            # Loss measures generator's ability to fool the discriminator
            loss_adv_ = loss_adv(discriminator(g_output_norm), valid)
            # contextual loss, VCG16 need 3 channels
            pred_3C = torch.cat((g_output_norm, g_output_norm, g_output_norm), dim=1)
            mask_3C = torch.cat((mask, mask, mask), dim=1)
            loss_con = Context_crit(pred_3C, mask_3C)

            g_loss = adv_ratio * loss_adv_  +  seg_ratio * loss_seg_ + con_ratio * loss_con

            g_loss.backward()
            optim_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optim_D.zero_grad()
            # detach()很重要，因为generator的梯度不需要传到discriminator!
            real_loss = loss_adv(discriminator(mask), valid) # 能不能区分出真实的mask 二分类交叉熵 BCELoss
            fake_loss = loss_adv(discriminator(g_output_norm.detach()), fake)  # 能不能区分出虚假的mask 二分类交叉熵 BCELoss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward(retain_graph=True)
            optim_D.step()

        ########### validation every epoch ############
        generator.eval()
        val_scores = []    
        with torch.no_grad():
            for i_batch_val, sample_batched in enumerate(dataloader_val):  # i_batch: steps
                img, mask = sample_batched['image'], sample_batched['mask']
                mask = mask.to(device).float() # ([8, 1, 512, 512])
                img = img.to(device) 
                g_output = generator(img) # ([8, 1, 512, 512])
                # g_output = tf(g_output)  #
                dice_cof, _ = metric_val(y_pred=g_output, y=mask) # y_pred自动经过sigmoid函数然后计算dice,实际部署中最后输出也应该经过sigmoid函数!!!!!!!!!!!!!!!!!!!!!!
                val_scores.append(dice_cof.cpu().numpy())

        metric = np.mean(val_scores) # update newest metric
        # print("mean dice score: ", metric)
        ######### validation every epoch ############

if __name__ == "__main__":
    # optuna search default using TPESampler
    # Only after some trials have been completed, Optuna uses the TPE algorithm to prune unpromising trials.
