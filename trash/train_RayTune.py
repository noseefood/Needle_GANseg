import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.ops.focal_loss import sigmoid_focal_loss
from dice_score import dice_coeff

from model.Generator import Generator 
from model.Discriminator import Discriminator
from util.read_data import SegmentationDataset

import contextual_loss as cl 

'''
bugs:
    无法兼容Monai/Torchmetric-> 修改为自定义的dice_coeff
'''


def train_seg(config):

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/imgs', help='input RGB or Gray image path')
    parser.add_argument('--mask_dir', type=str, default='C:\Research\projects\Learning\dataset\data_training\Data_Pork/masks', help='input mask path')
    parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--epoch", type=int, default=100, help="epoch in training")

    # parser.add_argument("--val_batch", type=int, default=200, help="Every val_batch, do validation")
    # parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

    args = parser.parse_args()

    # # os.makedirs('./save_model/save_G_Exp', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data setup
    dataset = SegmentationDataset(args.image_dir, args.mask_dir) 
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)]) # manual_seed fixed
    dataloader_train = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=config["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available())

    # Training setup
    generator = Generator().to(device)   # input channel must be 1
    discriminator = Discriminator().to(device) 
    
    # Define optimizer and loss function
    optim_G = torch.optim.Adam(generator.parameters(), lr=config["lr_G"], betas=(args.b1, args.b2))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr_D"], betas=(args.b1, args.b2))

    # loss_seg =  monai.losses.FocalLoss(alpha=0.75, gamma=2.0).to(device)
    # loss_seg = sigmoid_focal_loss(gamma=2.0,alpha=0.75,reduction='mean').to(device)
    loss_adv = torch.nn.BCELoss().to(device) 
    # metric_val = monai.metrics.DiceHelper(sigmoid=True)  
    # metric_val = dice_coeff().to(device)

    # tf = Compose([Activations(sigmoid=True)])
    Context_crit = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4',band_width=0.3).to(device) 
    batch_num = 0 
    
    for epoch in range(2):
        for i_batch, sample_batched in enumerate(dataloader_train):  # i_batch: steps
            # every mini-batch
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

            # loss_seg_ = loss_seg(input=g_output, target=mask) # focal loss自带sigmoid
            loss_seg_ = sigmoid_focal_loss(inputs=g_output, targets=mask, gamma=2.0,alpha=0.75,reduction='mean')

            # g_output_norm = tf(g_output) # ([8, 1, 256, 256])
            g_output_norm = torch.nn.functional.sigmoid(g_output)


            # Loss measures generator's ability to fool the discriminator
            loss_adv_ = loss_adv(discriminator(g_output_norm), valid)
            # contextual loss, VCG16 need 3 channels
            pred_3C = torch.cat((g_output_norm, g_output_norm, g_output_norm), dim=1)
            mask_3C = torch.cat((mask, mask, mask), dim=1)
            loss_con = Context_crit(pred_3C, mask_3C)

            g_loss = config["adv_ratio"] * loss_adv_  +  config["seg_ratio"] * loss_seg_ + config["con_ratio"] * loss_con

            g_loss.backward()
            optim_G.step()

            # print("loss_adv_", loss_adv_.item())
            # print("loss_seg_", loss_seg_.item())
            # print("loss_con", loss_con.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optim_D.zero_grad()

            real_loss = loss_adv(discriminator(mask), valid) # 能不能区分出真实的mask 二分类交叉熵 BCELoss
            # detach()很重要，因为generator的梯度不需要传到discriminator!
            fake_loss = loss_adv(discriminator(g_output_norm.detach()), fake)  # 能不能区分出虚假的mask 二分类交叉熵 BCELoss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(retain_graph=True)
            optim_D.step()

        ########## validation of generator ##########
        # validation after one full epoch
        generator.eval()
        val_scores = []    
        with torch.no_grad():
            for i_batch_val, sample_batched in enumerate(dataloader_val):  # i_batch: steps
                
                img, mask = sample_batched['image'], sample_batched['mask']
                mask = mask.to(device).float() # ([8, 1, 512, 512])
                img = img.to(device) 

                g_output = generator(img) # ([8, 1, 512, 512])
                # dice_cof, _ = metric_val(g_output, mask) # y_pred自动经过sigmoid函数然后计算dice,实际部署中最后输出也应该经过sigmoid函数!!!!!!!!!!!!!!!!!!!!!!
                dice_cof = dice_coeff(g_output, mask)
                val_scores.append(dice_cof.cpu().numpy())

        # print("val_scores", val_scores)
        metric = np.mean(val_scores)
        print("mean dice score: ", metric)

        # return metric for ray tune
        # train.report(mean_accuracy=metric)
        train.report({"accuracy": metric})
    
    print("Finished Training of ray tune")



def main(num_samples=1, max_num_epochs=2, gpus_per_trial=1):
    '''
    num_samples: number of trials from different hyperparameter configurations
    max_num_epochs: number of epochs to train for each trial
    gpus_per_trial: number of GPUs to allocate to each trial
    '''
    ray.init(local_mode=True)

    # Define search spaces
    search_space = {
        "lr_G": tune.loguniform(3e-4, 1e-2),
        "lr_D": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([4, 8]),
        "seg_ratio": tune.choice([0.7, 0.8, 0.9, 1.0]),
        "con_ratio": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "adv_ratio": tune.choice([0.1, 0.2, 0.3]),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_seg),
            resources={"cpu": 12, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
            # max_concurrent_trials = 2, # 2 trials run at the same time, it only for cluster
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    # print("Best trial final validation loss: {}".format(
    #     best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    
if __name__ == "__main__":
    main()




