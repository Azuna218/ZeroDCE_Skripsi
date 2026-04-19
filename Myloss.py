import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from torchvision.models import VGG16_Weights


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left  = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean     = torch.mean(org,     1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(
                org_pool - torch.FloatTensor([0.3]).cuda(),
                torch.FloatTensor([0]).cuda()
            ),
            torch.FloatTensor([0.5]).cuda()
        )
        E_1 = torch.mul(
            torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()),
            enhance_pool - org_pool
        )

        D_org_left  = F.conv2d(org_pool, self.weight_left,  padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up    = F.conv2d(org_pool, self.weight_up,    padding=1)
        D_org_down  = F.conv2d(org_pool, self.weight_down,  padding=1)

        D_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        D_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)

        D_left  = torch.pow(D_org_left  - D_enhance_left,  2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up    = torch.pow(D_org_up    - D_enhance_up,    2)
        D_down  = torch.pow(D_org_down  - D_enhance_down,  2)

        E = (D_left + D_right + D_up + D_down)
        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d    = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x        = x.size()[2]
        w_x        = x.size()[3]
        count_h    = (x.size()[2] - 1) * x.size()[3]
        count_w    = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:,  :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]  - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):

    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        r, g, b_ch = torch.split(x, 1, dim=1)
        mean_rgb    = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb  = torch.split(mean_rgb, 1, dim=1)
        Dr = r    - mr
        Dg = g    - mg
        Db = b_ch - mb
        k  = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        return torch.mean(k)


class perception_loss(nn.Module):

    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h             = self.to_relu_1_2(x)
        h_relu_1_2    = h
        h             = self.to_relu_2_2(h)
        h_relu_2_2    = h
        h             = self.to_relu_3_3(h)
        h_relu_3_3    = h
        h             = self.to_relu_4_3(h)
        h_relu_4_3    = h
        return h_relu_4_3


class L_SSIM(nn.Module):
    """
    Differentiable SSIM loss.
    Directly optimizes the SSIM evaluation metric during training.
    Loss = 1 - SSIM, so minimizing this = maximizing SSIM.
    """

    def __init__(self, window_size=11):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.window      = self._create_window(window_size)

    def _gaussian(self, size, sigma=1.5):
        coords = torch.arange(size).float() - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, size):
        _1d = self._gaussian(size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d

    def forward(self, img1, img2):
        _, c, _, _ = img1.shape
        window     = self.window.expand(c, 1, -1, -1).to(img1.device).to(img1.dtype)
        pad        = self.window_size // 2

        mu1    = F.conv2d(img1,      window, padding=pad, groups=c)
        mu2    = F.conv2d(img2,      window, padding=pad, groups=c)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return 1 - ssim_map.mean()