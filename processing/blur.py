import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomGaussianBlur(nn.Module):
    def __init__(self, prob_blur, size, sigma, n_channels=None):
        super(RandomGaussianBlur, self).__init__()
        self.prob_blur = prob_blur
        self.gaussian_blur = GaussianBlur(size, sigma, n_channels = n_channels)

    def forward(self, x):
        batch_size = x.size()[0]

        mask = torch.rand(batch_size) < self.prob_blur

        x[mask] = self.gaussian_blur(x[mask])
        return x


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma=None, same=True, n_channels=None, trainable=False):
        super(GaussianBlur, self).__init__()
        self.n_channels = n_channels or 3

        kernel_size = kernel_size or 3
        sigma = sigma or 0.5
        self.same = same
        if self.same:
            self.padding = nn.ReplicationPad2d((kernel_size - 1)//2).cuda()

        kernel = GaussianBlur.matlab_style_gauss2D(
            shape=(kernel_size, kernel_size),
            sigma=sigma
        )

        convolution_weight_numpy = np.stack([kernel[None, :, :] for _ in range(n_channels)])

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(torch.FloatTensor(convolution_weight_numpy))
        else:
            self.register_buffer('weight', torch.FloatTensor(convolution_weight_numpy))

    def forward(self, x):
        blurred = F.conv2d(
            self.padding(x) if self.same else x,
            self.weight if self.trainable else torch.autograd.Variable(self.weight),
            stride=1, padding=0, groups=self.n_channels)
        return blurred


    # From https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    # Not clear how this differs from other implementations
    @staticmethod
    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h