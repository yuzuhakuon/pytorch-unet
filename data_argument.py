import cv2
import os
from math import *
from random import choice
import random
import numpy as np
from collections import Counter
from glob import glob

width = 288
height = 464
provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
             "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
             "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
             "甘", "青", "宁", "新", "挂"]


def make_datasets():
    image_path = glob("data\\CCPD2020\\ccpd_green\\train\\*.jpg")
    for i, path in enumerate(image_path):
        if i % 100 == 0:
            print(i)
        path = path.strip()
        if path:
            name = path.split("\\")[-1]
            # 矩形四角坐标
            pos = path.split("-")[-4]
            pos = pos.split("_")
            pos = [xy.split("&") for xy in pos]
            pos = [[int(z) for z in xy] for xy in pos]
            assert len(pos) == 4

            # 左上与右下
            box = path.split("-")[-5]
            box = box.split("_")
            box = [xy.split("&") for xy in box]
            box = [[int(z) for z in xy] for xy in box]
            assert len(box) == 2

            # 中心与裁剪
            center_x = (box[0][0] + box[1][0])//2
            center_y = (box[0][1] + box[1][1])//2
            cut_x = center_x - width // 2
            cut_y = center_y - height // 2

            image = cv2.imread(path)
            gray = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(gray, np.array(pos), (255,))

            # input_ = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            # output = gray[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            input_ = image[cut_y:cut_y+height, cut_x:cut_x+width]
            output = gray[cut_y:cut_y + height, cut_x:cut_x + width]

            try:
                h, w = np.shape(output)
                assert h == 224, "高度不符合要求"
                assert w == 400, "宽度不符合要求"

                cv2.imwrite("data\\imgs\\{0}".format(name), input_)
                cv2.imwrite("data\\masks\\{0}".format(name), output)
            except:
                print("warning ********")


# 背景图片
bg_folder = glob(r"dataset\CCPD2019\ccpd_np\*.jpg")


# create random value between 0 and val-1
def r(val):
    return int(np.random.random() * val)


def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.7 + np.random.random() * 0.3)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.4 + np.random.random() * 0.6)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.4 + np.random.random() * 0.6)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def AddGauss(img, level=1):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


def rot(img, angel, shape, max_angel):
    """ 使图像轻微的畸变

        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸

    """
    size_o = [shape[1], shape[0]]

    size = (shape[1] + int(shape[0] * cos(
        (float(max_angel) / 180) * 3.14)), shape[0])

    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0],
                       [size_o[0], size_o[1]]])
    if (angel > 0):

        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0],
                           [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]],
                           [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0],
                       [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)],
                       [r(factor), shape[0] - r(factor)],
                       [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def dumpRotateImage(img, degree, pos):

    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)

    # 加入平移操作
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2

    imgRotation = cv2.warpAffine(img,
                                 matRotation, (widthNew, heightNew),
                                 borderValue=(255, 255, 255))
    pos_rot = []
    for P in pos:
        Q = np.dot(matRotation, np.array([[P[0]], [P[1]], [1]]))
        Q = [int(xy) for xy in Q]
        pos_rot.append(Q)
    return imgRotation, pos_rot


def resize(image, scale1, scale2):
    h, w, _ = image.shape
    new_h, new_w = int(h * scale1), int(w * scale2)
    image = cv2.resize(image, (new_w, new_h))

    return image


def AddNoiseSingleChannel(single):
    diff = 255-single.max()
    noise = np.random.normal(0, 1+r(6), single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise = diff*noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst


def addNoise(img, sdev=0.5, avg=10):
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0])
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1])
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2])
    return img


def gasuss_noise(image, sigma=30, factor=0.1):
    h, w, c = image.shape
    noise = np.random.normal(0, sigma, (h, w, c))
    mask = np.random.rand(h, w, c)
    noise[mask > factor] = 0
    image = np.uint8(image.astype(np.int) + noise)
    return image


def add_gaussian_noise(image_in, noise_sigma):
    """
        给图片添加高斯噪声
        image_in:输入图片
        noise_sigma：
        """
    temp_image = np.float64(np.copy(image_in))

    h, w, _ = temp_image.shape
    # 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return np.uint8(noisy_image)


def sp_noise(image, prob=0.01):
    h, w, c = image.shape
    mask = np.random.rand(h, w, 3)
    image[mask < prob] = 255

    return image


def pos_adjust(pos, h, w):
    for i, p in enumerate(pos):
        if p[0] < 0:
            pos[i][0] = 0
        if p[0] >= w:
            pos[i][1] = w - 1

        if p[1] < 0:
            pos[i][0] = 0
        if p[1] >= h:
            pos[i][1] = h - 1
    return pos


def transform(image):
    gasuss_up = 0
    rotate = False

    if (r(10) < 2):
        image = cv2.rotate(image, cv2.ROTATE_180)
        rotate = True

    # NOTE: 高斯噪声
    if (r(10) < 3):
        image = gasuss_noise(image)
        gasuss_up += 1
    # NOTE: 椒盐噪声
    if (r(10) < 3):
        image = sp_noise(image)
        gasuss_up += 1
    if (r(10) < 2):
        image = addNoise(image)

    if (r(10) < 2):
        image = tfactor(image)
    # NOTE: 高斯模糊
    if (r(10) < 1):
        level = r(1)
        image = AddGauss(image, level)

    # NOTE: 放缩
    if (r(10) < 3):
        scale = (r(50) + 50) / 70
        image = resize(image, 1, scale)
    if (r(10) < 3):
        scale = (r(50) + 50) / 70
        image = resize(image, scale, 1)
    if (r(10) < 7):
        scale = (r(100) + 100) / 150
        image = resize(image, scale, scale)

    p_h, p_w, p_c = image.shape
    pos = [[0, 0], [p_w, 0], [0, p_h], [p_w, p_h]]
    # NOTE: 随机旋转
    if (r(10) < 8):
        degree = 10 - r(20)
        image, pos = dumpRotateImage(image, degree, pos)

    return image, pos, rotate


def merge(fg, bg, pos=None):
    bg_h, bg_w, bg_c = bg.shape
    white_bg = np.ones_like(bg, dtype=np.uint8) * 255

    fg_h, fg_w, fg_c = fg.shape

    h = bg_h - fg_h - 1
    w = bg_w - fg_w - 1

    top_x = random.randint(0, w)
    top_y = random.randint(0, h)

    white_bg[top_y:top_y + fg_h, top_x:top_x + fg_w, :] = fg
    mask1 = ((white_bg[:, :, 0] == 255) & (white_bg[:, :, 1] == 255) &
             (white_bg[:, :, 2] == 255))
    mask1 = np.expand_dims(mask1, 2).repeat(3, axis=2)
    mask2 = mask1 == False
    image = bg * (mask1.astype(bg.dtype)) + white_bg * (mask2.astype(
        white_bg.dtype))
    return image, top_x, top_y


def pos2str(pos, top_x, top_y):
    pos_str = list()
    for p in pos:
        p[0] += top_x
        p[1] += top_y
        pos_str.append(str(p[0]) + "&" + str(p[1]))
    return "_".join(pos_str)


def main():
    image_folder = glob(
        r"E:\Code\github\chinese_license_plate_generator\multi_val\*.jpg")
    bg_folder = glob(r"C:\Users\Mashiro\Desktop\Image\bg\*.jpg")
    save_folder = r"E:\Code\project\Pytorch-UNet\data"

    for i, path in enumerate(image_folder):
        if i % 100 == 0:
            print(i)
        filename = path.split("\\")[-1]
        filename, _ = os.path.splitext(filename)
        pro2num = dict(zip(provinces, range(len(provinces))))
        plate_num, color, is_double = filename.split("-")
        plate_num = list(plate_num)
        plate_num[0] = str(pro2num[plate_num[0]])
        plate_num = "_".join(plate_num)

        try:
            bg_path = choice(bg_folder)
            bg = cv2.imread(bg_path)
            bg = resize(bg, 0.6, 0.6)

            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

            image = resize(image, 0.5, 0.5)
            h, w, c = image.shape
            size_ = (w, h)
            image, pos, rotate = transform(image)
            image, top_x, top_y = merge(image, bg)
            image = gasuss_noise(image, 30, 0.3)
            image = AddGauss(image, 1)
            pos_str = pos2str(pos, top_x, top_y)
            name = plate_num+"-"+color+"-"+is_double + \
                "-" + pos_str + "-" + str(rotate) + ".jpg"

            poly_pos = [pos[0], pos[1], pos[3], pos[2]]
            gray = np.zeros(bg.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(gray, np.array(poly_pos), (255,))
            cv2.imwrite(os.path.join(save_folder, "imgs", name), image)
            cv2.imwrite(os.path.join(save_folder, "masks", name), gray)
        except:
            print("warning")


vocabulary = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
              "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
              "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
              "Y", "Z"]


def cnn_process():
    image_folder = glob(
        r"E:\Code\github\chinese_license_plate_generator\reco_val\*.jpg")
    save_folder = r"E:\Code\temp\Pytorch-UNet\data\reco"

    for i, path in enumerate(image_folder):
        if i % 100 == 0:
            print(i)
        filename = path.split("\\")[-1]
        filename, _ = os.path.splitext(filename)
        pro2num = dict(zip(vocabulary, range(len(vocabulary))))
        plate_num, color, is_double = filename.split("-")
        plate_num = list(plate_num)
        plate_num = [str(pro2num[char]) for char in plate_num]
        plate_num = "_".join(plate_num)

        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            image = resize(image, 0.4, 0.4)
            if (r(10) < 5):
                image = gasuss_noise(image)
            if (r(10) < 5):
                image = sp_noise(image)
            if (r(10) < 5):
                image = tfactor(image)
            if (r(10) < 7):
                level = r(2)
                image = AddGauss(image, level)

            cv2.imwrite(os.path.join(save_folder, "val",
                                     plate_num + ".jpg"), image)
        except:
            print("warning")


def test():
    image_folder = glob(
        r"E:\Code\github\chinese_license_plate_generator\reco\*.jpg")

    for i, path in enumerate(image_folder):
        if i % 100 == 0:
            print(i)
        filename = path.split("\\")[-1]
        filename, _ = os.path.splitext(filename)
        pro2num = dict(zip(vocabulary, range(len(vocabulary))))
        plate_num, color, is_double = filename.split("-")
        plate_num = list(plate_num)
        plate_num = [str(pro2num[char]) for char in plate_num]
        plate_num = "_".join(plate_num)


if __name__ == "__main__":
    main()
