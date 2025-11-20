import numpy as np
import math
from .niqe_utils import calculate_niqe
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color, exposure
import torch
import cv2

"""URanker"""
def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return {"x": d_img_org, "x_his": x_his}

def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32

    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad

    img = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)

    return img

def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram


def getURanker(image: np.array, uranker_model):
    inputs = torch.from_numpy(image).float()
    inputs = inputs.permute(0, 3, 1, 2)  # B, H, W, C => B, C, H, W
    inputs = preprocessing(inputs)
    uiqa = 0.0
    with torch.no_grad():
        uiqa += torch.sum(
            uranker_model(**inputs)["final_result"].squeeze(-1).squeeze(-1)
        ).item()
    return uiqa

"""
UCIQE
======================================
https://ieeexplore.ieee.org/document/7300447
Compute the Underwater Color Image Quality Evaluation (UCIQE) score.

UCIQE is a metric for assessing the quality of underwater images based on
chroma, saturation, and luminance contrast.
======================================
"""

def get_uciqe(image):
    hsv = cv2.cvtColor(np.array(image * 255, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    mu = np.mean(S) / 255
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    Maxsum, Minsum = 0, 0
    V1, V2 = V / 255, V / 255

    for i in range(1, number + 1):
        Maxvalue = np.amax(np.amax(V1))
        x, y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0], y[0]]
        V1[x[0], y[0]] = 0

    top = Maxsum / number

    for i in range(1, number + 1):
        Minvalue = np.amin(np.amin(V2))
        X, Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0], Y[0]]
        V2[X[0], Y[0]] = 1

    bottom = Minsum / number

    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe


def getUCIQE(image):
    # image:  B, H, W, C

    UCIQE = 0
    for i in range(image.shape[0]):
        UCIQE += get_uciqe(image[i, :, :, :])
    return UCIQE


### NIQE ### 
def getNIQE(image):
    # image:  B, H, W, C
    NIQE = 0
    for i in range(image.shape[0]):
        NIQE += calculate_niqe(image[i, :, :, :][:, :, ::-1] * 255)
    return NIQE


##############################################################################
def getPSNR(img, imclean, data_range):
    # Img = img.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    # Iclean = imclean.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += peak_signal_noise_ratio(
            imclean[i, :, :, :], img[i, :, :, :], data_range=data_range
        )
    return PSNR


def getSSIM(img, imclean, data_range):
    # Img = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C
    # Iclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C

    SSIM = 0
    for i in range(img.shape[0]):
        SSIM += structural_similarity(
            imclean[i, :, :, :],
            img[i, :, :, :],
            data_range=data_range,
            channel_axis=-1,
            win_size=5,
        )
    return SSIM


class Evaluator:
    def __init__(self, no_ref=False, uranker_model=None):
        self.no_ref = no_ref
        self.uranker_model = uranker_model
        self.reset()

    def reset(
        self,
    ):
        if self.no_ref:
            self.niqe = 0.0
            self.uciqe = 0.0
            self.uranker = 0
        else:
            self.ssim = 0.0
            self.psnr = 0.0
        self.count = 0

    def evaluation(self, pred, label):
        if self.no_ref:
            self.niqe += getNIQE(pred)
            self.uciqe += getUCIQE(pred)
            self.uranker += getURanker(pred, self.uranker_model)
        else:
            self.psnr += getPSNR(pred, label, 1.0)
            self.ssim += getSSIM(pred, label, 1.0)
        self.count += pred.shape[0]

    def getMean(self):
        if self.no_ref:
            self.niqe /= self.count
            self.uciqe /= self.count
            self.uranker /= self.count
            return self.niqe, self.uciqe, self.uranker
        else:
            self.ssim /= self.count
            self.psnr /= self.count
            return self.ssim, self.psnr
