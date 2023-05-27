from torch import nn
from utils import CBR, DWConv, MLP, DWT_2D


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


class Feature_Extraction(nn.Module):
    """
    Feature Extraction Layer in SemLA
    Extraction of registration features and semantic awareness maps

    Args:
        'x' (torch.Tensor): (N, C, H, W)
        'train_mode' (String)
    """
    def __init__(self):
        super().__init__()

        # Discrete Wavelet Transform (For feature map downsampling)
        self.dwt = DWT_2D(wave='haar')

        self.reg0 = JConv(1, 8)
        self.reg1 = JConv(32, 16)
        self.reg2 = JConv(64, 32)
        self.reg3 = JConv(128, 256)
        self.pred_reg = nn.Sequential(JConv(256, 256), JConv(256, 256), JConv(256, 256))

        self.sa0 = JConv(256, 256)
        self.sa1 = JConv(256, 128)
        self.sa2 = JConv(128, 32)
        self.sa3 = JConv(32, 1)
        self.pred_sa = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode):
        x0 = self.reg0(x)
        x1 = self.reg1(self.dwt(x0))
        x2 = self.reg2(self.dwt(x1))
        x3 = self.reg3(self.dwt(x2))
        feat_reg = self.pred_reg(x3)

        # Training the registration of SemLA
        if mode == 'train_reg':
            return feat_reg

        # Training the semantic awareness of SemLA
        elif mode == 'train_sa':
            y0 = self.sa0(feat_reg)
            y1 = self.sa1(y0)
            y2 = self.sa2(y1)
            y3 = self.sa3(y2)
            feat_sa = self.pred_sa(y3)
            return feat_sa

        # Testing the registration and semantic awareness of SemLA (Other modules in SemLA are not included)
        elif mode == 'test':
            y0 = self.sa0(feat_reg)
            y1 = self.sa1(y0)
            y2 = self.sa2(y1)
            y3 = self.sa3(y2)
            feat_sa = self.pred_sa(y3)
            return feat_reg, feat_sa

