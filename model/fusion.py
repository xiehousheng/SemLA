from torch import nn
import torch
from model.utils import CBR, DWConv, MLP
import torch.nn.functional as F

class SemLA_Fusion(nn.Module):
    """
    The registration section of SemLA
    """

    def __init__(self):
        super().__init__()

        self.fuse1 = CR(1, 8)
        self.fuse2 = CR(8, 8)
        self.fuse3 = CR(8, 16)
        self.fuse4 = CR(16, 16)
        self.fuse5 = JConv(48, 1)
        self.acitve = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, matchmode = 'semantic'):
        bs = x.shape[0]

        feat1 = self.fuse1(x)
        feat2 = self.fuse2(feat1)
        feat3 = self.fuse3(feat2)
        feat4 = self.fuse4(feat3)

        featfuse=torch.cat((feat1,feat2,feat3,feat4),dim=1)

        (vifeat, irfeat) = featfuse.split(int(bs / 2))

        if matchmode == 'semantic':
            featfuse = irfeat * mask * 0.7 + vifeat * mask * 0.6 + vifeat * (1 - mask)
        elif matchmode == 'scene':
            featfuse = irfeat * 0.6 + vifeat * 0.6

        featfuse = self.fuse5(featfuse)
        featfuse = self.acitve(featfuse)
        featfuse = (featfuse + 1) / 2

        return featfuse


class JConv(nn.Module):
    """Joint Convolutional blocks

    Args:
        'x' (torch.Tensor): (N, C, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(JConv, self).__init__()
        self.feat_trans = CBR(in_channels, out_channels)
        self.dwconv = DWConv(out_channels)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.mlp = MLP(out_channels, bias=True)

    def forward(self, x):
        x = self.feat_trans(x)
        x = x + self.dwconv(x)
        out = self.norm(x)
        x = x + self.mlp(out)
        return x



class CR(nn.Module):
    """Convolution with Leaky ReLU

    Args:
        'x' (torch.Tensor): (N, C, H, W)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)