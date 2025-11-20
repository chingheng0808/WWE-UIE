import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


## HIN Block
class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super(BasicBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out = out + self.identity(x)
        return out


class WaveletEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)

        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)

        fea = self.fuse(dwt)  # -> [B, C, H//2, W//2]
        fea = self.post(fea)  # -> [B, C, H//2, W//2]

        out = F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)

        return out


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super(GetGradient, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "sobel":
            # sobel filter
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

            kernel_y = (
                torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            kernel_x = (
                torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = (
                torch.tensor(kernel_laplace, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.register_buffer(
                "kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1)
            )

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)

            grad_magnitude = torch.sqrt(
                torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-6
            )
        elif self.mode == "laplacian":
            grad_magnitude = F.conv2d(
                x, self.kernel_laplace, padding=1, groups=self.dim
            )
            grad_magnitude = torch.abs(grad_magnitude)  # magnitude only

        return grad_magnitude


class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super(SGFB, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.frdb2(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=48):
        super(BasicLayer, self).__init__()
        self.fwawb = WaveletEnhanceBlock(feature_channels)
        self.sgfb = SGFB(feature_channels)

    def forward(self, x):
        res = x
        x = self.fwawb(x) + x
        x = self.sgfb(x)
        return 0.5 * x + 0.5 * res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super(GrayWorldRetinex, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gray_mean = mean.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        gain = gray_mean / (mean + self.eps)
        x = x * gain  # white balance
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out


class myModel(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32, use_white_balance=False):
        super(myModel, self).__init__()
        self.use_white_balance = use_white_balance
        if self.use_white_balance:
            self.wb = GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.first = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, stride=1, padding=1
        )
        self.encoder1 = BasicLayer(feature_channels)
        self.down1 = Downsample(feature_channels)
        self.encoder2 = BasicLayer(feature_channels * 2**1)
        self.down2 = Downsample(feature_channels * 2**1)
        self.bottleneck = BasicLayer(feature_channels * 2**2)
        self.up1 = Upsample(feature_channels * 2**2)
        self.decoder1 = BasicLayer(feature_channels * 2**1)
        self.up2 = Upsample(feature_channels * 2**1)
        self.decoder2 = BasicLayer(feature_channels)
        self.out = nn.Conv2d(
            feature_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        res = x
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x
        x1 = self.first(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        x = self.up1(x3) + x2
        x = self.decoder1(x)
        x = self.up2(x) + x1
        x = self.decoder2(x)
        out = self.out(x) + res
        return out


if __name__ == "__main__":
    dummy_img = torch.rand(1, 3, 128, 128)
    model = myModel()
    output_img, output_img_ds, learned_weight_map = model(dummy_img)
    print("Output image shape:", output_img.shape)
    print("Learned weight map shape:", learned_weight_map.shape)
