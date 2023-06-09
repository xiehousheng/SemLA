import cv2
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms as tfs

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

class PilGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

def enhance(img0, img1, IMAGE_SHAPE):
    # The four vertices of the image
    src_point = np.array([[0, 0],
                          [IMAGE_SHAPE[1] - 1, 0],
                          [0, IMAGE_SHAPE[0] - 1],
                          [IMAGE_SHAPE[1] - 1, IMAGE_SHAPE[0] - 1]], dtype=np.float32)

    # Perspective Information
    dst_point = get_dst_point(0.1, IMAGE_SHAPE)

    # Rotation and scale transformation
    rotation = 40
    rot = random.randint(-rotation, rotation)
    scale = 1.2 + random.randint(-90, 100) * 0.01

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img0, out_img1 = cv2.warpPerspective(img0, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])),\
            cv2.warpPerspective(img1, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))


    return out_img0, out_img1, mat


class Dataset(Dataset):
    def __init__(self, img_vi_path, img_ir_path, label_path, train_size_w, train_size_h):
        super(Dataset, self).__init__()
        self.img_vi_path = img_vi_path
        self.img_ir_path = img_ir_path
        self.label_path = label_path

        self.trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        data_floder = os.listdir(self.img_vi_path)
        self.data_floder = data_floder

        self.train_size_w = int(train_size_w)
        self.train_size_h = int(train_size_h)

        self.feat_size_w = int(self.train_size_w / 8)
        self.feat_size_h = int(self.train_size_h / 8)

        self.feat_size_wh = int(self.feat_size_w * self.feat_size_h)

    def __getitem__(self, idx):
        item = self.data_floder[idx]

        imagevi, imageir = cv2.imread(os.path.join(self.img_vi_path, item), cv2.IMREAD_GRAYSCALE), cv2.imread(
            os.path.join(self.img_ir_path, item), cv2.IMREAD_GRAYSCALE)

        label = cv2.imread(os.path.join(self.label_path, item), cv2.IMREAD_GRAYSCALE)

        imagevi = cv2.resize(imagevi, (self.train_size_w, self.train_size_h))
        imageir = cv2.resize(imageir, (self.train_size_w, self.train_size_h))
        label = cv2.resize(label, (self.train_size_w, self.train_size_h))

        seed = random.random()
        if seed < 0.2:
            (h, w) = imagevi.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, random.randint(-10, 10), random.randint(3, 8) * 0.1)
            imagevi = cv2.warpAffine(imagevi, M, (w, h))
            imageir = cv2.warpAffine(imageir, M, (w, h))
            label = cv2.warpAffine(label, M, (w, h))

        seed = random.random()
        if seed < 0.25:
            imagevi = motion_blur(imagevi, degree=random.randint(5, 14), angle=random.randint(-45, 45))
        seed = random.random()
        if seed < 0.25:
            imageir = motion_blur(imageir, degree=random.randint(5, 14), angle=random.randint(-45, 45))
        seed = random.random()
        if seed < 0.25:
            imagevi = gasuss_noise(imagevi, mean=0, var=0.001)
        seed = random.random()
        if seed < 0.25:
            imageir = gasuss_noise(imageir, mean=0, var=0.001)

        label_trans, imageir, mat = enhance(label, imageir, [self.train_size_h, self.train_size_w])

        mask_idx = cv2.resize(label, (self.feat_size_w, self.feat_size_h))
        mask_idx = np.asarray(mask_idx).reshape(-1)
        mask_idx = np.where(mask_idx < 20)[0]

        mask_trans_idx = cv2.resize(label_trans, (self.feat_size_w, self.feat_size_h))
        mask_trans_idx = np.asarray(mask_trans_idx).reshape(-1)
        mask_trans_idx = np.where(mask_trans_idx < 200)[0]

        imagevi = Image.fromarray(imagevi)
        imageir = Image.fromarray(imageir)

        seed = random.random()
        if seed < 0.25:
            imagevi = imagevi.filter(PilGaussianBlur(radius=random.randint(1, 2)))

        imagevi = tfs.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1)(imagevi)
        imageir = tfs.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.1)(imageir)

        imagevi = np.asarray(imagevi)
        imageir = np.asarray(imageir)


        point1 = np.array(list(range(0, self.feat_size_wh)))
        point1 = 8 * np.stack([point1 % self.feat_size_w, point1 // self.feat_size_w], axis=1).reshape(1, -1, 2).astype(np.float32)

        point2 = cv2.perspectiveTransform(point1, mat).reshape(-1, 2)
        point1 = point1.reshape(-1, 2)

        mask0 = np.where(point1[:, 0] < self.train_size_w - 8, True, False) * np.where(point1[:, 0] > 8, True, False) \
                * np.where(point1[:, 1] > 8, True, False) * np.where(point1[:, 1] < self.train_size_h - 8, True, False)

        mask1 = np.where(point2[:, 0] < self.train_size_w - 8, True, False) * np.where(point2[:, 0] > 8, True, False) \
                * np.where(point2[:, 1] > 8, True, False) * np.where(point2[:, 1] < self.train_size_h - 8, True, False)

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

        gt_conf_matrix = np.zeros([self.feat_size_wh, self.feat_size_wh], dtype=float)
        gt_conf_matrix[mkpts0, mkpts1] = 1.0
        gt = gt_conf_matrix

        gt[mask_idx, :] = 0
        gt[:, mask_trans_idx] = 0

        imagevi, imageir = transforms.ToPILImage()(imagevi), transforms.ToPILImage()(imageir)



        return self.trans(imagevi), self.trans(imageir), gt_conf_matrix, gt

    def __len__(self):
        return len(self.data_floder)


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


def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out