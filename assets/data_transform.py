import os

from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
import torch.nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import random
from torchvision import transforms as tfs
import matplotlib
import matplotlib.pyplot as plt




def enhance(img):
    IMAGE_SHAPE = [240,320]

    src_point = np.array([[               0,                0],
                          [IMAGE_SHAPE[1]-1,                0],
                          [               0, IMAGE_SHAPE[0]-1],
                          [IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1]], dtype=np.float32) 

    dst_point = get_dst_point(0.1, IMAGE_SHAPE)  

    # rot = random.randint(-2, 2) * config['homographic']['rotation'] + random.randint(0, 15)  
    rotation = 30
    rot = random.randint(-rotation, rotation)

    rot = random.randint(-30, 30)

    while rot >= -10 and rot <= 10:
        rot = random.randint(-30, 30)


   

    # sc = random.randint(-50, 50)
    #
    # if sc >= -20 and sc <= 20:
    #     sc = random.randint(-30, 30)


    # scale = 1.2 - config['homographic']['scale'] * random.random()
    scale = 1.0 +  random.randint(-50, 50) * 0.01  

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img = cv2.warpPerspective(img, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

    return out_img, mat


def get_dst_point(perspective, IMAGE_SHAPE):
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*a
        left_bottom_y  = 0.9 + perspective*e
        right_bottom_x = 0.9 + perspective*c
        right_bottom_y = 0.9 + perspective*f
    else:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*e
        left_bottom_y  = 0.9 + perspective*b
        right_bottom_x = 0.9 + perspective*f
        right_bottom_y = 0.9 + perspective*d

    dst_point = np.array([(IMAGE_SHAPE[1]*left_top_x,IMAGE_SHAPE[0]*left_top_y,1),
            (IMAGE_SHAPE[1]*right_top_x, IMAGE_SHAPE[0]*right_top_y,1),
            (IMAGE_SHAPE[1]*left_bottom_x,IMAGE_SHAPE[0]*left_bottom_y,1),
            (IMAGE_SHAPE[1]*right_bottom_x,IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    return dst_point

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color=None,
        kpts0=None, kpts1=None, text=[], dpi=100, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1,2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')

    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=(124/255,252/255,0), linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=(124/255,252/255,0), s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=(124/255,252/255,0), s=4)

    # put txts
    # txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

class PilGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

def gasuss_noise(image, mean=0, var=0.001):
   
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out

def motion_blur(image, degree=15, angle=45):
    image = np.array(image)

   
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
# if __name__ == '__main__':
#     image_path = os.listdir(r'D:\SLANet\dataset\MSRS\ir')
#     for image_name in image_path:
#         image=cv2.imread(r'D:\SLANet\dataset\MSRS\ir'+'\\'+image_name)
#         image=cv2.resize(image,(320,240))
#         cv2.imwrite(r'D:\SLANet\dataset\MSRS\irr'+'\\'+image_name,image)


if __name__ == '__main__':
    image_path = os.listdir(r'D:\SLANet\dataset\m3fd\vii')
    for image_name in image_path:
        image=cv2.imread(r'D:\SLANet\dataset\m3fd\transir'+'\\'+image_name)
        image=cv2.resize(image,(320,240))
        cv2.imwrite(r'D:\SLANet\dataset\m3fd\tranir'+'\\'+image_name,image)

        # image1 = cv2.imread(r'D:\SLANet\dataset\Ir' + '\\' + image_name)
        # image1 = cv2.resize(image1, (320, 240))
        # cv2.imwrite(r'D:\SLANet\dataset\m3fd\irr' + '\\' + image_name, image1)




if __name__ == '__main__':
    image_path = os.listdir(r'D:\SLANet\dataset\m3fd\vii')
    for image_name in image_path:
        # image = cv2.imread(r'D:\SLANet\dataset\MSRS\vii' + '\\' + image_name)
        # image = gasuss_noise(image, mean=0, var=0.001)
        # cv2.imwrite(r'D:\SLANet\dataset\MSRS\vii' + '\\' + image_name, image)

        image=cv2.imread(r'D:\SLANet\dataset\m3fd\irr'+'\\'+image_name)
        # image=cv2.resize(image,(320,240))
        out_img, mat=enhance(image)
        # out_img = gasuss_noise(out_img, mean=0, var=0.001)
        cv2.imwrite(r'D:\SLANet\dataset\m3fd\transir'+'\\'+image_name,out_img)
        np.save(r'D:\SLANet\dataset\m3fd\mat'+'\\' + image_name + '.npy', mat)

# #
#
#
#
#
#
#
#
#
#

#         point1 = np.array(([[290, 100], [223, 320], [4, 12], [9, 16]])).reshape(1, -1, 2).astype(np.float32)
#         point2 = cv2.perspectiveTransform(point1, mat).reshape(-1, 2)
#
# if __name__ == '__main__':
#     image_path = os.listdir(r'D:\SLANet\dataset\MSRS\irr')
#     for image_name in image_path:
#         ir = cv2.imread(r'D:\SLANet\dataset\MSRS\transir' + '\\' + image_name, cv2.IMREAD_GRAYSCALE)
#         vi = cv2.imread(r'D:\SLANet\dataset\MSRS\vii' + '\\' + image_name, cv2.IMREAD_GRAYSCALE)
#
#         mat = np.load(r'D:\SLANet\dataset\MSRS\mat'+'\\' + image_name + '.npy')
#
#         point1 = np.array(([[210, 100], [223, 150], [4, 12], [9, 16]])).reshape(1, -1, 2).astype(np.float32)
#         point2 = cv2.perspectiveTransform(point1, mat).reshape(-1, 2)
#         point1=point1.reshape(-1, 2)
#
#         fig = make_matching_figure(vi, ir, point1, point2)
#
#         # fuse=ir*0.5+vi*0.5
#         # plt.imshow(fuse,cmap='gray')
#
#         plt.show()

# if __name__ == '__main__':
#     image_path = os.listdir(r'D:\SLANet\dataset\MSRS\vi')
#     for image_name in image_path:
#         image = cv2.imread(r'D:\SLANet\dataset\MSRS\MSRS-master\crop_LR_visible' + '\\' + image_name)
#         image = cv2.resize(image, (320, 240))
#         cv2.imwrite(r'D:\SLANet\dataset\MSRS\vii' + '\\' + image_name, image)






