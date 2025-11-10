import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg19 import VGG19FeatureExtractor
import math
from math import exp
from torchvision.models import vgg16


# --- Perceptual loss network  --- #
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss) / len(loss)


class VGG19Loss(nn.Module):
    def __init__(self, layers_weights=None, device="cuda", loss_type="l1"):
        super(VGG19Loss, self).__init__()
        if layers_weights is None:
            layers_weights = {
                "relu2_2": 1.0,
                "relu3_4": 1.0,
                "relu4_4": 1.0,
                "relu5_4": 1.0,
            }
        self.layers_weights = layers_weights
        self.feature_extractor = VGG19FeatureExtractor(
            layers=tuple(layers_weights.keys())
        )
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "l2":
            self.criterion = nn.MSELoss()
        self.feature_extractor.to(device)

    def forward(self, input_img, target_img):
        """
        :param input_img: Tensor, shape [B, 3, H, W], in range [0,1]
        :param target_img: Tensor, shape [B, 3, H, W], in range [0,1]
        :return: perceptual loss (scalar)
        """
        input_features = self.feature_extractor(input_img)
        target_features = self.feature_extractor(target_img)
        loss = 0.0
        for layer, weight in self.layers_weights.items():
            loss += weight * self.criterion(
                input_features[layer], target_features[layer]
            )
        return loss


####### SSIM Loss Function #######
# The following code implements the SSIM (Structural Similarity Index) loss function.
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    eps_val = 1e-8

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
        L = L if L > eps_val else eps_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2 + eps_val
    v2 = sigma1_sq + sigma2_sq + C2 + eps_val
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        channel=3,
        size_average=True,
        val_range=None,
        device="cuda",
    ):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = channel
        self.window = create_window(window_size, channel).to(device).type(torch.float32)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        return ssim(
            img1,
            img2,
            window=self.window,
            window_size=self.window_size,
            size_average=self.size_average,
        )


class SSIMLoss(nn.Module):
    def __init__(
        self, window_size=11, size_average=True, val_range=1, channel=3, device="cuda"
    ):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(
            window_size=window_size,
            channel=channel,
            size_average=size_average,
            val_range=val_range,
            device=device,
        )

    def forward(self, img1, img2):
        ssim_ = self.ssim(img1, img2)
        return 1 - ssim_


###### Edge-aware Loss Function ######
class EdgeAwareLoss(nn.Module):
    def __init__(self, loss_type="l1", device="cuda"):
        super(EdgeAwareLoss, self).__init__()
        self.loss_type = loss_type.lower()

        self.sobel_kernel_x = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_kernel_y = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_kernel_x = self.sobel_kernel_x.repeat(3, 1, 1, 1).to(device)
        self.sobel_kernel_y = self.sobel_kernel_y.repeat(3, 1, 1, 1).to(device)
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type: choose either 'l1' or 'l2'")

    def forward(self, pred, gt):
        B, C, H, W = pred.shape

        sobel_x = self.sobel_kernel_x.repeat(C, 1, 1, 1)  # shape: [C, 1, 3, 3]
        sobel_y = self.sobel_kernel_y.repeat(C, 1, 1, 1)  # shape: [C, 1, 3, 3]

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=C)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=C)
        gt_edge_x = F.conv2d(gt, sobel_x, padding=1, groups=C)
        gt_edge_y = F.conv2d(gt, sobel_y, padding=1, groups=C)

        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-6)

        return self.loss_fn(pred_edge, gt_edge)


############# L1 Charbonnierloss ##############
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
