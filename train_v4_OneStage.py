#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_v3.py
@Time    :   2023/12/31
@Author  :   Xuesong Li
@Version :   4.0
@Contact :   xuesosng.li@tum.de
@Feature :   Using diffferent training model(1. Only using Focal and Contextual loss; 2. Using Focal, Contextual and Adverserial loss)
'''
# Only using Focal and Contextual loss to train the model
import json
import argparse
import os
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from util.read_data_v3 import SegmentationDataset

from model.Generator import Generator 
from model.Discriminator_v3 import Discriminator # third version

import numpy as np
import monai

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

import contextual_loss as cl
import random

torch.manual_seed(777)
np.random.seed(777)
random.seed(777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loops(args, dataset, generator, optim_G, loss_seg, metric_val, scheduler_G):
    # split train and val dataset
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)]) # manual_seed fixed

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # define tensorboard writer
    writer = SummaryWriter()
    batch_num = 0 

    '''lower band_width value would make the similarity function sharper, making the loss more sensitive 
    to differences between features. In contrast, a higher band_width value would make the similarity function 
    smoother, making the loss less sensitive to differences between features.'''

    args_dict = args.__dict__
    writer.add_hparams(args_dict, {})

    post_processing = Compose([Activations(sigmoid=True)])

    best_metric = -100 # best metric for all trials
    best_metric_batch = -1 # best metric 

    # train loop
    for epoch in range(args.epoch):
        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            batch_num += 1 
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            
            mask = mask.to(device).float()
            img = img.to(device) 

            generator.train()  # recover to train mode(because of eval in validation)
            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()

            g_output = generator(img) 

            loss_seg_ = loss_seg(input=g_output, target=mask) # focal loss自带sigmoid

            g_loss = loss_seg_ 

            # g_loss.backward(retain_graph=True) # 详见 https://blog.csdn.net/qxqsunshine/article/details/82973979
            g_loss.backward()
            optim_G.step()

            print("loss_seg_", loss_seg_.item())


            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader_train), g_loss.item())
            )

            # tensorboard log
            writer.add_scalar('G_loss in First Stage', g_loss.item(), epoch * len(dataloader_train) + i_batch)

            # write lr into tensorboard
            writer.add_scalar('lr_G', optim_G.param_groups[0]['lr'], epoch * len(dataloader_train) + i_batch)


            ########## validation of generator ##########
            if batch_num % (args.val_batch) == 0:

                # visualize the input, mask and output
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')

                # validation
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

                print("val_scores", val_scores)
                metric = np.mean(val_scores)
                print("mean dice score: ", metric)

                writer.add_scalar("val_mean_dice", metric, epoch * len(dataloader_train) + i_batch)

                # update the best model for all trials
                if metric > best_metric:
                    best_metric = metric
                    best_metric_batch = batch_num
                    torch.save(generator.state_dict(), f"./save_model/best_in{best_metric_batch}_FirstStage.pth")
                    print("Current best metric: ", metric)

        scheduler_G.step() # learning rate decay count

                

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./data/Basic_Pork/imgs', help='input RGB or Gray image path')
parser.add_argument('--mask_dir', type=str, default='./data/Basic_Pork/masks', help='input mask path')
parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')

parser.add_argument('--lrG', type=float, default='3e-4', help='learning rate')
parser.add_argument('--optimizer', type=str, default='Adam', help='RMSprop/Adam/SGD')
parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--epoch", type=int, default=100, help="epoch in training")

parser.add_argument("--val_batch", type=int, default=100, help="Every val_batch, do validation")
parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

parser.add_argument("--seg_ratio", type=float, default=1, help="Ratio of seg loss in generator loss") # 0.3

args = parser.parse_args()
print('args', args)

os.makedirs('./save_model', exist_ok=True)

with open('./save_model/arg_FirstStage.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


dataset = SegmentationDataset(args.image_dir, args.mask_dir, resolution=512)  # 512*512 

generator = Generator().to(device)   
discriminator = Discriminator().to(device) 

# define optimizer
if args.optimizer == "RMSprop":
    optim_G = torch.optim.RMSprop(generator.parameters(), lr = args.lrG)
elif args.optimizer == "Adam": 
    optim_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))
elif args.optimizer == "SGD":
    optim_G = torch.optim.SGD(generator.parameters(), lr=args.lrG, momentum=0.9)
    

############# learning rate decay #############
# scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=4, gamma=0.5) # step_size 
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optim_G, milestones=[8, 16, 32], gamma=0.5) # step_size
###############################################

# define loss
loss_adv = torch.nn.BCELoss().to(device) # GAN adverserial loss
metric_val = monai.metrics.DiceHelper(sigmoid=True)  # DICE score for validation of generator 最终输出的时候也应该经过sigmoid函数!!!!!!!!!!!!!!!!!!!!!!
loss_seg =  monai.losses.FocalLoss(alpha=0.75, gamma=2.0).to(device) # FocalLoss is an extension of BCEWithLogitsLoss, so sigmoid is not needed.


train_loops(args, dataset, generator, optim_G, loss_seg, metric_val, scheduler_G)
#!!!!!!!!!!!!!1 FocalLoss的参数不要用默认值(alpha=0.2)，否则根本无法训练