import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from Extractor import Feature_Extraction

class BCELoss(nn.Module):
    """cross-entropy loss function

    Args:
        'inputs': (torch.Tensor): (N, C, H, W)
        'targets': (torch.Tensor): (N, C, H, W)
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.BCE_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.BCE_loss(inputs, targets)

        # weighting loss
        weight = torch.zeros_like(targets, dtype=torch.float32)
        weight = weight.fill_(0.9)
        weight[targets > 0] = 1.7
        bce_loss = torch.mean(bce_loss * weight)

        return bce_loss


class Loss_stage1(nn.Module):
    """Calculate the loss of the first stage of training, including registration loss and semantic awareness loss

    Args:
        'inputs': (torch.Tensor): (N, C, H, W)
        'targets': (torch.Tensor): (N, C, H, W)
    """
    def __init__(self):
        super().__init__()
        self.backbone = Feature_Extraction()
        self.bceloss = BCELoss()

    def forward(self, reg_vi, reg_ir, conf_gt, sa_vi, sa_ir, sa_vi_gt, sa_ir_gt):
        bs_reg = reg_vi.shape[0]
        feat_reg = self.backbone(torch.cat((reg_vi, reg_ir), dim=0), mode='train_reg')
        (feat_reg_vi, feat_reg_ir) = feat_reg.split(bs_reg)

        feat_reg_vi = rearrange(feat_reg_vi, 'n c h w -> n (h w) c')
        feat_reg_ir = rearrange(feat_reg_ir, 'n c h w -> n (h w) c')

        feat_reg_vi, feat_reg_ir = map(lambda feat: feat / feat.shape[-1] ** .5,
                                       [feat_reg_vi, feat_reg_ir])

        # The registration loss is implemented based on the code of [LoFTR](https://github.com/zju3dv/LoFTR)
        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi,
                            feat_reg_ir) / 0.1

        # dual-softmax operator
        conf = F.softmax(conf, 1) * F.softmax(conf, 2)
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)

        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0

        alpha = 0.25
        gamma = 2.0

        pos_conf = conf[pos_mask]
        loss_reg = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

        # registration loss
        loss_reg = loss_reg.mean()
        bs_sa = sa_vi.shape[0]


        feat_sa = self.backbone(torch.cat((sa_vi, sa_ir), dim=0), mode='train_sa')
        (feat_sa_vi, feat_sa_ir) = feat_sa.split(bs_sa)

        loss_sa_vi = self.bceloss(feat_sa_vi, sa_vi_gt)
        loss_sa_ir = self.bceloss(feat_sa_ir, sa_ir_gt)

        # semantic awareness loss
        loss_sa = loss_sa_vi + loss_sa_ir

        return loss_reg, loss_sa, conf

