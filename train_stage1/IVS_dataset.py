import cv2
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


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


def enhance(img, label, img2, label2, IMAGE_SHAPE):
    # The four vertices of the image
    src_point = np.array([[0, 0],
                          [IMAGE_SHAPE[1] - 1, 0],
                          [0, IMAGE_SHAPE[0] - 1],
                          [IMAGE_SHAPE[1] - 1, IMAGE_SHAPE[0] - 1]], dtype=np.float32)

    # Perspective Information
    dst_point = get_dst_point(0.2, IMAGE_SHAPE)

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
    out_img, out_img2 = cv2.warpPerspective(img, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])), cv2.warpPerspective(label, mat,
                                                                                                             (
                                                                                                                 IMAGE_SHAPE[
                                                                                                                     1],
                                                                                                                 IMAGE_SHAPE[
                                                                                                                     0]))

    out_img3, out_img4 = cv2.warpPerspective(img2, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])), cv2.warpPerspective(label2,
                                                                                                               mat,
                                                                                                               (
                                                                                                                   IMAGE_SHAPE[
                                                                                                                       1],
                                                                                                                   IMAGE_SHAPE[
                                                                                                                       0]))

    return out_img, out_img2, out_img3, out_img4


class IVSDataset(Dataset):
    def __init__(self, img_vi_path, img_ir_path, label_path, train_size_w, train_size_h):
        super(IVSDataset, self).__init__()
        self.img_vi_path = img_vi_path
        self.img_ir_path = img_ir_path
        self.label_path = label_path

        self.trans1 = transforms.Compose([
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.trans2 = transforms.Compose([
            transforms.ColorJitter(brightness=0., contrast=0.1, saturation=0., hue=0.),
            transforms.ToTensor()
        ])

        data_floder = os.listdir(self.img_vi_path)
        self.data_floder = data_floder

        self.train_size_w = int(train_size_w)
        self.train_size_h = int(train_size_h)

        self.feat_size_w = int(self.train_size_w / 8)
        self.feat_size_h = int(self.train_size_h / 8)

    def __getitem__(self, idx):
        item = self.data_floder[idx]

        imagevi, labelvi = cv2.imread(os.path.join(self.img_vi_path, item)), cv2.imread(
            os.path.join(self.label_path, item), cv2.IMREAD_GRAYSCALE)

        imageir, labelir = cv2.imread(os.path.join(self.img_ir_path, item)), cv2.imread(
            os.path.join(self.label_path, item), cv2.IMREAD_GRAYSCALE)

        imagevi = cv2.resize(imagevi, (self.train_size_w, self.train_size_h))
        labelvi = cv2.resize(labelvi, (self.train_size_w, self.train_size_h))
        imageir = cv2.resize(imageir, (self.train_size_w, self.train_size_h))
        labelir = cv2.resize(labelir, (self.train_size_w, self.train_size_h))

        seed = random.random()
        if seed < 0.25:
            imagevi = motion_blur(imagevi, degree=random.randint(6, 15), angle=random.randint(-45, 45))
        seed = random.random()
        if seed < 0.25:
            imageir = motion_blur(imageir, degree=random.randint(6, 15), angle=random.randint(-45, 45))
        seed = random.random()
        if seed < 0.25:
            imagevi = gasuss_noise(imagevi, mean=0, var=0.001)
        seed = random.random()
        if seed < 0.25:
            imageir = gasuss_noise(imageir, mean=0, var=0.001)

        imagevi, labelvi, imageir, labelir = enhance(imagevi, labelvi, imageir, labelir,
                                                     [self.train_size_h, self.train_size_w])

        imagevi, labelvi = transforms.ToPILImage()(imagevi), transforms.ToPILImage()(
            cv2.resize(labelvi, (self.feat_size_w, self.feat_size_h)))
        imageir, labelir = transforms.ToPILImage()(imageir), transforms.ToPILImage()(
            cv2.resize(labelir, (self.feat_size_w, self.feat_size_h)))

        return self.trans1(imagevi), self.trans2(labelvi), self.trans1(imageir), self.trans2(labelir)

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