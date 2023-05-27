import torch
import torch.nn as nn
from teacher.model.backbone import build_backbone
from teacher.model.backbone.fine_preprocess import FinePreprocess
from teacher.model.backbone.coarse_matching import CoarseMatching
from teacher.model.backbone.fine_matching import FineMatching
from einops.einops import rearrange


class Matchformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.backbone = build_backbone()
        self.coarse_matching = CoarseMatching(config['matchformer']['match_coarse'])

    
    
    def forward(self, data):
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            
        if data['hw0_i'] == data['hw1_i']:
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1),(feat_f0, feat_f1) = feats_c.split(data['bs']),feats_f.split(data['bs'])
        else:
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        # match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

        