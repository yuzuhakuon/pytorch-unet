import numpy as np
import cv2
import os
import argparse
import logging


import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet2 import Unet
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):

    return (mask * 255).astype(np.uint8)


def image_by_mask(image, mask):
    pass


def show_box(image, imgray):
    ret, thresh = cv2.threshold(
        imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 大津阈值
    # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        # 最小外接矩形框，有方向角
        rect = cv2.minAreaRect(cnt)
        # cv2.cv.Boxpoints()  # if imutils.is_cv2()else cv2.boxPoints(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    return image


if __name__ == "__main__":

    img_path = "5.jpg"

    net = Unet(3, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(
        'checkpoints\\CP_epoch40.pth', map_location=device))

    origin = cv2.imread(img_path)
    img = cv2.resize(origin, (256, 256))
    img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_ = Image.open(img_path)
    img = img_.resize((256, 256))
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=1,
                       out_threshold=0.5,
                       device=device)
    result = mask_to_image(mask)
    print(result.shape)
    result = cv2.resize(result, (600, 337))

    image = show_box(origin, result)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
