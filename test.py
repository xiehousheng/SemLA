import torch
import cv2
from model.SemLA import SemLA
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange
from model.utils import YCbCr2RGB, RGB2YCrCb, make_matching_figure
import os

# Test on dataset
if __name__ == '__main__':
    # config
    vi_path = ""
    ir_path = ""
    result_path = ""
    reg_weight_path = ""
    fusion_weight_path = ""
    match_mode  = 'scene' # 'semantic' or 'scene'

    matcher = SemLA()
    # Loading the weights of the registration model
    matcher.load_state_dict(torch.load(reg_weight_path), strict=False)

    # Loading the weights of the fusion model
    matcher.load_state_dict(torch.load(fusion_weight_path), strict=False)

    matcher = matcher.eval().cuda()

    image_path = os.listdir(vi_path)
    # vi_path is the visible dataset path, ir_path is the infrared dataset path
    for image_name in image_path:
        img0_pth = os.path.join(vi_path, image_name)
        img1_pth = os.path.join(ir_path, image_name)

        img0_raw = cv2.imread(img0_pth)
        img0_raw = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        img0_raw = cv2.resize(img0_raw, (320, 240))  # input size shuold be divisible by 8
        img1_raw = cv2.resize(img1_raw, (320, 240))

        img0 = torch.from_numpy(img0_raw)[None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

        img0 = rearrange(img0, 'n h w c ->  n c h w')
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

        mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir = matcher(vi_Y, img1, matchmode=match_mode)
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

        _, prediction = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5)
        prediction = np.array(prediction, dtype=bool).reshape([-1])
        mkpts0_tps = mkpts0[prediction]
        mkpts1_tps = mkpts1[prediction]
        tps = cv2.createThinPlateSplineShapeTransformer()
        mkpts0_tps = mkpts0_tps.reshape(1, -1, 2)
        mkpts1_tps = mkpts1_tps.reshape(1, -1, 2)

        matches = []
        for j in range(1, mkpts0.shape[0] + 1):
            matches.append(cv2.DMatch(j, j, 0))

        tps.estimateTransformation(mkpts0_tps, mkpts1_tps, matches)
        img1_raw_trans = tps.warpImage(img1_raw)
        sa_ir = tps.warpImage(sa_ir[0][0].detach().cpu().numpy())
        sa_ir = torch.from_numpy(sa_ir)[None][None].cuda()

        img1_trans = torch.from_numpy(img1_raw_trans)[None][None].cuda() / 255.
        fuse = matcher.fusion(torch.cat((vi_Y, img1_trans), dim=0), sa_ir, matchmode=match_mode).detach()

        fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)
        fuse = fuse.detach().cpu()[0]
        fuse = rearrange(fuse, ' c h w ->  h w c').detach().cpu().numpy()

        fig = make_matching_figure(fuse, img0_raw, img1_raw, mkpts0, mkpts1, path=os.path.join(result_path, image_name))





