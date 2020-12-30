import torch
from torch import nn
from torch import optim
import cv2
import numpy as np
import torch.nn.functional as F


class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        #x1 = F.pad(x1, [(x2.size()[3] - x1.size()[3]) // 2, (x2.size()[3] - x1.size()[3]) - (x2.size()[3] - x1.size()[3]) // 2,
        #                (x2.size()[2] - x1.size()[2]) // 2, (x2.size()[2] - x1.size()[2]) - (x2.size()[2] - x1.size()[2]) // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Unet, self).__init__()

        self.n_classes = out_ch
        self.n_channels = in_ch
        self.bilinear = bilinear

        base_depth = 16
        self.conv1 = DoubleConv2(in_ch, base_depth)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2(base_depth, base_depth*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2(base_depth*2, base_depth*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2(base_depth*4, base_depth*8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv2(base_depth * 8, base_depth * 16)

        #self.up6 = nn.ConvTranspose2d(base_depth*16, base_depth*8, 2, stride=2)
        #self.conv6 = DoubleConv2(base_depth * 16, base_depth * 8)
        self.up6 = Up(base_depth * 16, base_depth * 8)

        # self.up7 = nn.ConvTranspose2d(base_depth*8, base_depth*4, 2, stride=2)
        # self.conv7 = DoubleConv2(base_depth * 8, base_depth * 4)
        self.up7 = Up(base_depth * 8, base_depth * 4)

        # self.up8 = nn.ConvTranspose2d(base_depth*4, base_depth*2, 2, stride=2)
        # self.conv8 = DoubleConv2(base_depth * 4, base_depth * 2)
        self.up8 = Up(base_depth * 4, base_depth * 2)

        # self.up9 = nn.ConvTranspose2d(base_depth*2, base_depth, 2, stride=2)
        # self.conv9 = DoubleConv2(base_depth * 2, base_depth)
        self.up9 = Up(base_depth * 2, base_depth)

        self.conv10 = nn.Conv2d(base_depth, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)
        x = self.up6(c5, c4)

        # up_7 = self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        x = self.up7(x, c3)


        # up_8 = self.up8(c7)
        # merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        x = self.up8(x, c2)


        # up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, c1], dim=1)
        # c9 = self.conv9(merge9)
        x = self.up9(x, c1)


        c10 = self.conv10(x)
        #out = nn.Sigmoid()(c10)
        return c10


def train():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    imgs = torch.randn(1, 3, 224, 400).to(device)

    net = Unet(3, 1)
    net.to(device=device)
    optimizer = optim.RMSprop(
        net.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    net.train()
    out = net(imgs)
    print(out.detach().numpy().shape)

    torch.save(net.state_dict(), 'epoch.pth')


def to_onnx():
    img_path = "3.jpg"

    net = Unet(3, 1)
    device = torch.device('cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(
        'epoch.pth', map_location=device))

    img = torch.randn(1, 3, 360, 360).to(device)

    net.eval()
    img = img.to(device=device, dtype=torch.float32)

    export_onnx_file = "unet2.onnx"			# 目的ONNX文件名
    torch.onnx.export(net, img, export_onnx_file,
                      opset_version=11, verbose=True)


def test_onnx():
    model = cv2.dnn.readNetFromONNX("unet2.onnx")


if __name__ == "__main__":
    to_onnx()
    test_onnx()
