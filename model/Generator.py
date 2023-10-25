import torch
import torch.nn as nn
from model.Attention import ChannelAttentionModule, SpatialAttentionModule

'''
Reference: CBAM: Convolutional Block Attention Module
Specialization: 
    1. TODO: Using nn.LeakyReLU instead of nn.ReLU : OK
    2.
'''

class Generator(nn.Module):

    # Unet的skip connections + spatial attentio + channel attention

    def __init__(self, in_features=1, out_features=1, init_features=32, LeakyReLU=False):
        super(Generator, self).__init__()

        ReLU = None
        if LeakyReLU:
            ReLU = nn.LeakyReLU(0.2, inplace=True)
        else:
            ReLU = nn.ReLU()

        features = init_features
        
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),  # (512-3+2)/1+1=512 keep size ((N+2P-K)/S)+1, ((N+2P-F)/S)向下取整
            torch.nn.BatchNorm2d(num_features=features),
            ReLU,
        )

        # Attention block to refine the features
            # spatial attention: 就是对于所有的通道，在二维平面上，对H x W尺寸的特征图学习到一个权重，对每个像素都会学习到一个权重。你可以想象成一个像素是C维的一个向量，深度是C，在C个维度上，权重都是一样的，但是在平面上，权重不一样
            # Channel Attention：就是对每个C（通道），在channel维度上，学习到不同的权重，平面维度上权重相同。所以基于通道域的注意力通常是对一个通道内的信息直接全局平均池化，而忽略每一个通道内的局部信息
        self.channelAttModul = ChannelAttentionModule(features)
        self.spatialAttModul = SpatialAttentionModule() # 

        self.encode_layer1_half = torch.nn.Sequential(  # 
            torch.nn.Conv2d(in_channels=features,
                            out_channels=features,
                            kernel_size=3, # 一般为奇数,最好还是3,经典参数
                            padding=1, # 一般为kernel_size-1,可以保证size
                            stride=1), # 一般为1,保证size;放大时为2可以有效减少图片的size
            torch.nn.BatchNorm2d(num_features=features),
            ReLU,
            )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        def UpDownConv_block(in_features, out_features):
            layers = [
                torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, stride=1),
                torch.nn.BatchNorm2d(num_features=out_features),
                ReLU,
                torch.nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1, stride=1),
                torch.nn.BatchNorm2d(num_features=out_features),
                ReLU,
            ]
            block = nn.Sequential(*layers)
        
            return block

        self.encode_layer2 = UpDownConv_block(features, features * 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.encode_layer3 = UpDownConv_block(features * 2, features * 4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.encode_layer4 = UpDownConv_block(features * 4, features * 8)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)



        self.encode_decode_layer = UpDownConv_block(features * 8, features * 16)



        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = UpDownConv_block(features * 16, features * 8)


        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = UpDownConv_block(features * 8, features * 4)


        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = UpDownConv_block(features * 4, features * 2)


        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = UpDownConv_block(features * 2, features)

        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )


    def forward(self, x):
        # Skip connection similar to UNet Structure
        '''
        args: x-[8(batchsize), 1, 512, 512]
        return: out-[8(batchsize), 1, 512, 512]
        '''
        enc1 = self.encode_layer1(x)

        ca = self.channelAttModul(enc1) * enc1
        sa = self.spatialAttModul(ca) * ca

        enc1_half = self.encode_layer1_half(sa)
        enc2 = self.encode_layer2(self.pool1(enc1_half))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        # TODO : extarct features from bottleneck
        

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # skip connection similar to UNet 
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1) 
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)


        return out
