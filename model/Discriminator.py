import torch.nn as nn

'''
Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py
           https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py  (original version from this repo)
Specialization: 
1. Using nn.LeakyReLU instead of nn.ReLU
2. BatchNorm2d used in discriminator
'''


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        

        def discriminator_block(in_filters, out_filters, normalize=True, dropout=True):

            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]

            if dropout:
                block.append(nn.Dropout2d(0.25))

            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, normalize=False),  # Only single channel Gray scale img
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image(final dense layer)
        ds_size = 256 // 2 ** 4 # 256 is the original size of image(after resize)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()) # Sigmoid necessary?


    def forward(self, img):
        '''
        args: img-[8(batchsize), 1, 512, 512]
        return: validity-[8(batchsize), 1]  Real or Fake result
        ''' 
        out = self.model(img) # torch.Size([8, 128, 32, 32])
        out = out.view(out.shape[0], -1) # torch.Size([8, 131072])
        validity = self.adv_layer(out)  #torch.Size([batchsize, 1]) 

        return validity
