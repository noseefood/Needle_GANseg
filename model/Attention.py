import torch
import torch.nn as nn

'''
Code Reference: 
    https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    https://arxiv.org/pdf/2004.03696.pdf
Specialization: in Attebtion Module, using nn.ReLU instead of nn.LeakyReLU!!!, because the expected ouput should be in [0, 1] range.
'''

class ChannelAttentionModule(nn.Module):
    '''
    Classical channel attention module, which is used to generate channel attention map
    Paper: CBAM(Convolutional Block Attention Module) 
    Principle: 
    '''
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 1×1×C 
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 1×1×C

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x)) 
        maxout = self.shared_MLP(self.max_pool(x))

        return self.sigmoid(avgout + maxout) # 1×1×C


class SpatialAttentionModule(nn.Module):
    '''
    Classical spatial attention module, which is used to generate spatial attention map
    Paper: CBAM(Convolutional Block Attention Module)
    Principle: cat(avg_pool(x), max_pool(x)) -> conv2d -> sigmoid -> spatial attention map
    '''
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid() # 一个Sigmoid激活函数，用于将卷积层的输出压缩到0到1之间(相当于关注度概率)，以产生空间注意力图

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True) # 相当于将将前输入的所有channel压缩成一个channel
        maxout, _ = torch.max(x, dim=1, keepdim=True) # 也相当于将将前输入的所有channel压缩成一个channel
        out = torch.cat([avgout, maxout], dim=1) # chanbel=2
        out = self.sigmoid(self.conv2d(out)) # 通过Sigmoid输出一个channel的空间注意力图(0-1)
        
        return out
