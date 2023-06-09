import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
from torchvision import transforms as tfs

"""
Implementation based on the modification of 
https://github.com/791136190/UnsuperPoint_PyTorch/blob/main/Unsuper/dataset/coco.py
"""

def get_dst_point(perspective, IMAGE_SHAPE):
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = perspective * a
        left_top_y = perspective * b
        right_top_x = 0.9 + perspective * c
        right_top_y = perspective * d
        left_bottom_x = perspective * a
        left_bottom_y = 0.9 + perspective * e
        right_bottom_x = 0.9 + perspective * c
        right_bottom_y = 0.9 + perspective * f
    else:
        left_top_x = perspective * a
        left_top_y = perspective * b
        right_top_x = 0.9 + perspective * c
        right_top_y = perspective * d
        left_bottom_x = perspective * e
        left_bottom_y = 0.9 + perspective * b
        right_bottom_x = 0.9 + perspective * f
        right_bottom_y = 0.9 + perspective * d

    dst_point = np.array([(IMAGE_SHAPE[1] * left_top_x, IMAGE_SHAPE[0] * left_top_y, 1),
                          (IMAGE_SHAPE[1] * right_top_x, IMAGE_SHAPE[0] * right_top_y, 1),
                          (IMAGE_SHAPE[1] * left_bottom_x, IMAGE_SHAPE[0] * left_bottom_y, 1),
                          (IMAGE_SHAPE[1] * right_bottom_x, IMAGE_SHAPE[0] * right_bottom_y, 1)], dtype='float32')
    return dst_point


def enhance(img, IMAGE_SHAPE):
    # The four vertices of the image
    src_point = np.array([[0, 0],
                          [IMAGE_SHAPE[1] - 1, 0],
                          [0, IMAGE_SHAPE[0] - 1],
                          [IMAGE_SHAPE[1] - 1, IMAGE_SHAPE[0] - 1]], dtype=np.float32)

    # Perspective Information
    dst_point = get_dst_point(0.2, IMAGE_SHAPE)

    # Rotation and scale transformation
    rotation = 25
    rot = random.randint(-rotation, rotation)
    scale = 1.2 + random.randint(-90, 100) * 0.01

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img = cv2.warpPerspective(img, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

    return out_img, mat, f_point


class PilGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)


def motion_blur(image, degree=15, angle=45):
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def resize_img(img, IMAGE_SHAPE):
    h, w = img.shape[:2]
    if h < IMAGE_SHAPE[0] or w < IMAGE_SHAPE[1]:
        new_h = IMAGE_SHAPE[0]
        new_w = IMAGE_SHAPE[1]
        h = new_h
        w = new_w
        img = cv2.resize(img, (new_w, new_h))
    new_h, new_w = IMAGE_SHAPE
    try:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
    except:
        print(h, new_h, w, new_w)
        raise
    if len(img.shape) == 2:
        img = img[top: top + new_h, left: left + new_w]  # crop image
    else:
        img = img[top: top + new_h, left: left + new_w, :]
    return img


def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out


class RegDataset(Dataset):
    def __init__(self, data_path_vi, data_path_ir, train_size_w, train_size_h):
        super(RegDataset, self).__init__()

        self.trans = transforms.Compose([
            transforms.ToTensor()
        ])

        self.data_floder = os.listdir(data_path_vi)
        # data = []
        # for item in data_floder:
        #     data.append((os.path.join(data_path_vi, item)))
        self.data_path_vi = data_path_vi
        self.data_path_ir = data_path_ir

        self.train_size_w = int(train_size_w)
        self.train_size_h = int(train_size_h)
        self.num_of_points = int((self.train_size_w / 8) * (self.train_size_h / 8))

        self.feat_size_w = self.train_size_w/8
        self.feat_size_h = self.train_size_h/8
      



    def __getitem__(self, idx):
        image_name = self.data_floder[idx]


        # visible images, obtained from COCO dataset
        image_vi = cv2.imread(os.path.join(self.data_path_vi, image_name), cv2.IMREAD_GRAYSCALE)
        image_vi = cv2.resize(image_vi, (self.train_size_w, self.train_size_h))

        # pseudo-infrared image, obtained by CPSTN
        # refer to the Details of Implementation section of the SemLA paper
        image_ir = cv2.imread(os.path.join(self.data_path_ir, image_name[:-3]+'png'), cv2.IMREAD_GRAYSCALE)
        image_ir = cv2.resize(image_ir, (self.train_size_w, self.train_size_h))


        # data enhancement
        seed = random.random()
        if seed < 0.25:
            (h, w) = image_vi.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, random.randint(-20, 20), random.randint(10,40) * 0.1)
            image_vi = cv2.warpAffine(image_vi, M, (w, h))
            image_ir = cv2.warpAffine(image_ir, M, (w, h))

        seed = random.random()
        if seed < 0.25:
            image_vi = motion_blur(image_vi, degree=random.randint(7, 13), angle=random.randint(-45, 45))
        seed = random.random()
        if seed < 0.25:
            image_ir = motion_blur(image_ir, degree=random.randint(7, 13), angle=random.randint(-45, 45))

        seed = random.random()
        if seed < 0.25:
            image_vi = gasuss_noise(image_vi, mean=0, var=0.001)
        seed = random.random()
        if seed < 0.25:
            image_ir = gasuss_noise(image_ir, mean=0, var=0.001)

        image_vi = resize_img(image_vi, [self.train_size_h, self.train_size_w])
        image_ir = resize_img(image_ir, [self.train_size_h, self.train_size_w])  # reshape the image
        image_ir, mat, f_point = enhance(image_ir, [self.train_size_h, self.train_size_w])


        # cv2 -> PIL
        image_vi = Image.fromarray(image_vi)  # rgb
        image_ir = Image.fromarray(image_ir)  # rgb

        seed = random.random()
        if seed < 0.25:
            image_vi = image_vi.filter(PilGaussianBlur(radius=random.randint(1, 2)))

        image_vi = tfs.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1)(image_vi)

        seed = random.random()
        if seed < 0.25:
            image_ir = image_ir.filter(PilGaussianBlur(radius=random.randint(1, 2)))

        image_ir = tfs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image_ir)

        # PIL -> cv2
        image_vi = np.asarray(image_vi)
        image_ir = np.asarray(image_ir)


        # Generate ground truth confidence matrix based on the synthesized transformation matrix
        point1 = np.array(list(range(0,  self.num_of_points)))
        point1 = 8 * np.stack([point1 % self.feat_size_w, point1 // self.feat_size_w], axis=1).reshape(1, -1, 2).astype(np.float32)

        point2 = cv2.perspectiveTransform(point1, mat).reshape(-1, 2)
        point1 = point1.reshape(-1, 2)

        mask0 = np.where(point1[:, 0] < (self.train_size_w-8), True, False) * np.where(point1[:, 0] > 8, True, False) \
                * np.where(point1[:, 1] > 8, True, False) * np.where(point1[:, 1] < (self.train_size_h-8), True, False)

        mask1 = np.where(point2[:, 0] < (self.train_size_w-8), True, False) * np.where(point2[:, 0] > 8, True, False) \
                * np.where(point2[:, 1] > 8, True, False) * np.where(point2[:, 1] < (self.train_size_h-8), True, False)

        mask1 = mask0 * mask1
        point1 = point1.astype(np.int32)
        point2 = point2.astype(np.int32)
        point1 = point1[mask1]
        point2 = point2[mask1]

        point1 = point1 // 8
        point2 = point2 // 8

        mask = np.where(
            np.all((point2[:, :2] >= (0, 0)) & (point2[:, :2] < (self.feat_size_w, self.feat_size_h)),
                   axis=1))
        point1 = point1[mask][:, :2]
        point2 = point2[mask][:, :2]

        mkpts0 = point1[:, 1] * self.feat_size_w + point1[:, 0]
        mkpts1 = point2[:, 1] * self.feat_size_w + point2[:, 0]

        gt_conf_matrix = np.zeros([self.num_of_points, self.num_of_points], dtype=float)
        gt_conf_matrix[mkpts0.astype(np.int32), mkpts1.astype(np.int32)] = 1.0

        image_vi = transforms.ToPILImage()(image_vi)
        image_ir = transforms.ToPILImage()(image_ir)

        return (self.trans(image_vi), self.trans(image_ir), gt_conf_matrix)

    def __len__(self):
        return len(self.data_floder)
