from scipy import ndimage
from PIL import Image
import numpy as np
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import torch

'''URanker'''
def preprocessing(d_img_org):     
        d_img_org = padding_img(d_img_org)
        x_his = build_historgram(d_img_org)
        return {
            'x': d_img_org,
            'x_his': x_his
        }
        
def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32
    
    left_pad = (w_out- w) // 2
    right_pad = w_out - w - left_pad
    top_pad  = (h_out - h) // 2
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
    uiqa = 0.
    with torch.no_grad():
        uiqa += torch.sum(uranker_model(**inputs)['final_result'].squeeze(-1).squeeze(-1)).item()
    return uiqa

"""
UCIQE
======================================
https://ieeexplore.ieee.org/document/7300447
Compute the Underwater Color Image Quality Evaluation (UCIQE) score.

UCIQE is a metric for assessing the quality of underwater images based on
chroma, saturation, and luminance contrast.

implemented from: https://github.com/paulwong16/UCIQE, which is a matlab repository, we converted it to python
======================================
"""

def rgb2lab_n(f: np.ndarray):
    f = f.astype(np.float64)
    if f.max() > 1.0:
        f /= 255.0

    fr, fg, fb = f[..., 0], f[..., 1], f[..., 2]
    # linearize sRGB
    thr = 0.04045
    r_lin = np.where(fr <= thr, fr / 12.92, ((fr + 0.055) / 1.055) ** 2.4)
    g_lin = np.where(fg <= thr, fg / 12.92, ((fg + 0.055) / 1.055) ** 2.4)
    b_lin = np.where(fb <= thr, fb / 12.92, ((fb + 0.055) / 1.055) ** 2.4)

    # Convert to CIE XYZ (D65)
    x = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin
    y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin

    # Normalize by reference white
    xn, yn, zn = 0.9505, 1.0, 1.0891
    p = x / xn
    q = y / yn
    rr = z / zn

    # f(t) for Lab conversion
    eps = 0.008856
    f_p = np.where(p > eps, np.cbrt(p), 7.787 * p + 16.0 / 116.0)
    f_q = np.where(q > eps, np.cbrt(q), 7.787 * q + 16.0 / 116.0)
    f_r = np.where(rr > eps, np.cbrt(rr), 7.787 * rr + 16.0 / 116.0)

    # Compute L, a, b
    L = np.where(q > eps, 116.0 * np.cbrt(q) - 16.0, 903.3 * q)
    a = 500.0 * (f_p - f_q)
    b = 200.0 * (f_q - f_r)

    return L, a, b

def rgb_to_saturation(img: np.ndarray):
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    mx = np.max(img, axis=2)
    mn = np.min(img, axis=2)
    diff = mx - mn
    S = np.zeros_like(mx)
    mask = mx > 0
    S[mask] = diff[mask] / mx[mask]
    return S

def get_uciqe(img: np.ndarray) -> float:
    # Input image in RGB format. The pixel values should be in the range [0, 255] or normalized to [0, 1].
    # Ensure float in [0,1]
    im = img.astype(np.float64)
    if im.max() > 1.0:
        im /= 255.0

    # Convert to Lab and compute Chroma
    L, a, b = rgb2lab_n(im)
    Chroma = np.sqrt(a**2 + b**2)
    std_chroma = np.std(Chroma)

    # Compute mean saturation
    S = rgb_to_saturation(im)
    mean_sat = np.mean(S)

    # Compute contrast of luminance
    contrast_lum = np.max(L) - np.min(L)

    # UCIQE formula
    score = 0.4680 * std_chroma + 0.2745 * contrast_lum + 0.2576 * mean_sat
    return score

def getUCIQE(image):
    # image:  B, H, W, C
    
    UCIQE = 0
    for i in range(image.shape[0]):
        UCIQE += get_uciqe(image[i, :, :, :])
    return UCIQE

"""
   Computes the Underwater Image Quality Measure (UIQM)
   metrics paper: https://ieeexplore.ieee.org/document/7305804
   referenced from  https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py
"""

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # calculate mu_alpha weight
    weight = (1 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = 2. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0)
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log( (max_ / min_)+ 1e-6)
    return w * val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]

    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    # get eme for each channel
    r_eme = eme(R_edge_map, 8)
    g_eme = eme(G_edge_map, 8)
    b_eme = eme(B_edge_map, 8)

    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def plip_g(x, mu=1026.0):
    return mu - x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / (gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    # return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """

    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0

    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = -1. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    # entropy scale - higher helps with randomness
    alpha = 1

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)

            top = max_ - min_
            bot = max_ + min_

            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)

            # try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w * val


def get_uiqm(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    # x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    # c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm

def getUIQM(image):
    UIQM = 0
    for i in range(image.shape[0]):
        UIQM += get_uiqm(image[i, :, :, :])
    return UIQM


##############################################################################

def getPSNR(img, imclean, data_range):
    # Img = img.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    # Iclean = imclean.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += peak_signal_noise_ratio(imclean[i, :, :, :], img[i, :, :, :], data_range=data_range)
    return PSNR

def getSSIM(img, imclean, data_range):
    # Img = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C
    # Iclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C
    
    SSIM = 0
    for i in range(img.shape[0]):
        SSIM += structural_similarity(imclean[i,:,:,:], img[i,:,:,:],  data_range=data_range, channel_axis=-1,win_size=5)
    return SSIM

class Evaluator():
    def __init__(self, no_ref=False, uranker_model=None):
        self.no_ref = no_ref
        self.uranker_model = uranker_model
        self.reset()
    def reset(self,):
        if self.no_ref:
            self.uiqm = 0.
            self.uciqe = 0.
            self.uranker = 0
        else:
            self.ssim = 0.
            self.psnr = 0.
        self.count = 0 
    def evaluation(self, pred, label):
        if self.no_ref:
            self.uiqm += getUIQM(pred)
            self.uciqe += getUCIQE(pred)
            self.uranker += getURanker(pred, self.uranker_model)
        else:
            self.psnr += getPSNR(pred, label, 1.)
            self.ssim += getSSIM(pred, label, 1.)
        self.count += pred.shape[0]
    def getMean(self):
        if self.no_ref:
            self.uiqm /= self.count
            self.uciqe /= self.count
            self.uranker /= self.count
            return self.uiqm, self.uciqe, self.uranker
        else:
            self.ssim /= self.count
            self.psnr /= self.count
            return self.ssim, self.psnr