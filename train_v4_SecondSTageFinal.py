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

import contextual_loss as cl
import random

'''
First stage using Hybrid Loss (Dice + CL)
Second stage using Hybrid Loss (Dice + CL) + adversial loss
'''

torch.manual_seed(777)
np.random.seed(777)
random.seed(777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loops(args, dataset, generator, discriminator, 
                optim_G, optim_D, loss_adv, loss_seg, metric_val, scheduler_G):
    # split train and val dataset
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)]) # manual_seed fixed

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # define tensorboard writer
    writer = SummaryWriter()
    batch_num = 0 

    Context_crit = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4',band_width=0.8).to(device) 

    args_dict = args.__dict__
    writer.add_hparams(args_dict, {})

    best_metric = -100 # best metric for all trials
    best_metric_batch = -1 # best metric 

    # generator load pretrained model that only using focal and contextual loss
    generator.load_state_dict(torch.load('./best_metric_model_Generator0.87.pth')) # 

    # train loop
                  
    for epoch in range(args.epoch):
      
        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps

            generator.train()
            discriminator.train()
            batch_num += 1 
          
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)  # for discriminator 1为真   
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False) # for discriminator 0为假
            
            mask = mask.to(device).float()
            img = img.to(device) 

            # -----------------
            #  Train Generator
            # -----------------
            optim_G.zero_grad()
            g_output = generator(img) 
            g_pred = torch.sigmoid(g_output)

            loss_seg_ = loss_seg(input=g_pred, target=mask) # dice shut down the sigmoid
          
            # Loss measures generator's ability to fool the discriminator
            loss_adv_ = loss_adv(discriminator(g_pred), valid)
          
            # contextual loss, VCG16 need 3 channels
            pred_3C = torch.cat((g_pred, g_pred, g_pred), dim=1)
            mask_3C = torch.cat((mask, mask, mask), dim=1)
          
            loss_con = Context_crit(pred_3C, mask_3C)

            g_loss = args.adv_ratio * loss_adv_  +  args.seg_ratio * loss_seg_ + args.con_ratio * loss_con

            # g_loss.backward(retain_graph=True) # 详见 https://blog.csdn.net/qxqsunshine/article/details/82973979
          
            g_loss.backward()
            optim_G.step()

            print("loss_adv_", loss_adv_.item())
            print("loss_seg_", loss_seg_.item())
            print("loss_con", loss_con.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optim_D.zero_grad()

            real_loss = loss_adv(discriminator(mask), valid) # 能不能区分出真实的mask 二分类交叉熵 BCELoss
            # detach()很重要，因为generator的梯度不需要传到discriminator!
            fake_loss = loss_adv(discriminator(g_output_norm.detach()), fake)  # 能不能区分出虚假的mask 二分类交叉熵 BCELoss
            d_loss = real_loss + fake_loss

            # d_loss.backward(retain_graph=True) # 详见 https://blog.csdn.net/qxqsunshine/article/details/82973979
          
            d_loss.backward()
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader_train), d_loss.item(), g_loss.item())
            )

            # tensorboard log
            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader_train) + i_batch)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader_train) + i_batch)

            # write lr into tensorboard
            writer.add_scalar('lr_G', optim_G.param_groups[0]['lr'], epoch * len(dataloader_train) + i_batch)


          
            if batch_num % 150 == 0:
                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(g_output, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')



            ########## validation of generator ##########
            if batch_num % (args.val_batch) == 0:

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

                writer.add_scalar("val_mean_dice", metric, batch_num)
                print('metric:', metric)
              
                # save best metric model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_batch = batch_num
                    torch.save(model.state_dict(), './save_model/best_metric_model_Generator' + str(round(metric, 2)) +'.pth')
                    print('saved new best metric model')
                else:
                    print('not saved new best metric model')

            # if batch_num % (args.save_batch) == 0:
            #     torch.save(generator.state_dict(), f"./save_model/model_in{batch_num}.pth")

        scheduler_G.step() # learning rate decay count


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./data/Basic_Pork/imgs', help='input RGB or Gray image path')
parser.add_argument('--mask_dir', type=str, default='./data/Basic_Pork/masks', help='input mask path')
parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')

parser.add_argument('--lrG', type=float, default='3e-5', help='learning rate')
parser.add_argument('--lrD', type=float, default='1e-5', help='learning rate') # 
parser.add_argument('--optimizer', type=str, default='Adam', help='RMSprop/Adam/SGD')
parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--epoch", type=int, default=100, help="epoch in training")

parser.add_argument("--val_batch", type=int, default=100, help="Every val_batch, do validation")
parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

parser.add_argument("--adv_ratio", type=float, default=0.002, help="Ratio of adverserial loss in generator loss") # 0.7
parser.add_argument("--seg_ratio", type=float, default=1, help="Ratio of seg loss in generator loss") # 0.3
parser.add_argument("--con_ratio", type=float, default=0.001, help="Ratio of contextual loss in generator loss") # 0.2


args = parser.parse_args()
print('args', args)

os.makedirs('./save_model', exist_ok=True)

with open('./save_model/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


dataset = SegmentationDataset(args.image_dir, args.mask_dir, resolution=512)  # 512*512 

generator = Generator().to(device)   
discriminator = Discriminator().to(device) 

# define optimizer
if args.optimizer == "RMSprop":
    optim_D = torch.optim.RMSprop(discriminator.parameters(), lr = args.lrD)
    optim_G = torch.optim.RMSprop(generator.parameters(), lr = args.lrG)
elif args.optimizer == "Adam": 
    optim_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.b1, args.b2))
elif args.optimizer == "SGD":
    optim_G = torch.optim.SGD(generator.parameters(), lr=args.lrG, momentum=0.9)
    optim_D = torch.optim.SGD(discriminator.parameters(), lr=args.lrD, momentum=0.9)
    

############# learning rate decay #############
# scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=4, gamma=0.5) # step_size 
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optim_G, milestones=[12, 24, 48], gamma=0.5) # step_size
###############################################

# define loss
loss_adv = torch.nn.BCELoss().to(device) # GAN adverserial loss
metric_val = monai.metrics.DiceHelper(sigmoid=True)  # # sigmoid + 0.5 threshold
loss_seg =  monai.losses.Dice(sigmoid=False).to(device)



train_loops(args, dataset, generator, discriminator, optim_G, optim_D, loss_adv, loss_seg, metric_val, scheduler_G)
#!!!!!!!!!!!!!1 FocalLoss的参数不要用默认值(alpha=0.2)，否则根本无法训练
