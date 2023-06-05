import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from Reg import SemLA_Reg

class Loss_stage2(nn.Module):
    """Calculate the loss of the first stage of training, including registration loss and semantic awareness loss

    Args:
        'inputs': (torch.Tensor): (N, C, H, W)
        'targets': (torch.Tensor): (N, C, H, W)
    """
    def __init__(self):
        super().__init__()
        self.backbone = SemLA_Reg()

    def forward(self, img_vi, img_ir, conf_gt, str_conf_gt):
        feat_reg_vi_final, feat_reg_ir_final, feat_reg_vi_str, feat_reg_ir_str = self.backbone(torch.cat((img_vi, img_ir), dim=0))

        feat_reg_vi_final = rearrange(feat_reg_vi_final, 'n c h w -> n (h w) c')
        feat_reg_ir_final = rearrange(feat_reg_ir_final, 'n c h w -> n (h w) c')

        feat_reg_vi_final, feat_reg_ir_final = map(lambda feat: feat / feat.shape[-1] ** .5,
                                       [feat_reg_vi_final, feat_reg_ir_final])

        # The registration loss is implemented based on the code of [LoFTR](https://github.com/zju3dv/LoFTR)
        conf_0 = torch.einsum("nlc,nsc->nls", feat_reg_vi_final,
                            feat_reg_ir_final) / 0.1

        # dual-softmax operator
        conf_0 = F.softmax(conf_0, 1) * F.softmax(conf_0, 2)
        conf_0 = torch.clamp(conf_0, 1e-6, 1 - 1e-6)

        pos_mask_0, neg_mask_0 = conf_gt == 1, conf_gt == 0
        alpha = 0.25
        gamma = 2.0

        pos_conf_0 = conf_0[pos_mask_0]
        loss_0 = - alpha * torch.pow(1 - pos_conf_0, gamma) * pos_conf_0.log()

        # registration loss
        loss_0 = loss_0.mean()
        
        

        feat_reg_vi_str = rearrange(feat_reg_vi_str, 'n c h w -> n (h w) c')
        feat_reg_ir_str = rearrange(feat_reg_ir_str, 'n c h w -> n (h w) c')

        feat_reg_vi_str, feat_reg_ir_str = map(lambda feat: feat / feat.shape[-1] ** .5,
                                   [feat_reg_vi_str, feat_reg_ir_str])
        conf_1 = torch.einsum("nlc,nsc->nls", feat_reg_vi_str,
                              feat_reg_ir_str) / 0.1
        conf_1 = F.softmax(conf_1, 1) * F.softmax(conf_1, 2)
        conf_1 = torch.clamp(conf_1, 1e-6, 1 - 1e-6)

        pos_mask_1, neg_mask_1 = str_conf_gt == 1, str_conf_gt == 0
        pos_conf_1 = conf_1[pos_mask_1]

        loss_1 = - alpha * torch.pow(1 - pos_conf_1, gamma) * pos_conf_1.log()
        loss_1 = loss_1.mean()
 

    
        return loss_0, loss_1

