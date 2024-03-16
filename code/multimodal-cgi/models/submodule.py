from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class UniformSampler(nn.Module):
    def __init__(self):
        super(UniformSampler, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_samples=10):
        """
        Args:
            :min_disparity: lower bound of disparity search range
            :max_disparity: upper bound of disparity range predictor
            :number_of_samples (default:10): number of samples to be genearted.
        Returns:
            :sampled_disparities: Uniformly generated disparity samples from the input search range.
        """

        device = min_disparity.get_device()

        multiplier = (max_disparity - min_disparity) / (number_of_samples + 1)   # B,1,H,W
        range_multiplier = torch.arange(1.0, number_of_samples + 1, 1, device=device).view(number_of_samples, 1, 1)  #(number_of_samples, 1, 1)
        sampled_disparities = min_disparity + multiplier * range_multiplier

        print(sampled_disparities)

        return sampled_disparities


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and the disparity samples, generates:
                    - Warped right image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples

        Returns:
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """

        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_samples = disparity_samples.float()

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)

        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(
            right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())

        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) +
                                         (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
            (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        return warped_right_feature_map, left_feature_map




def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def groupwise_difference(fea1, fea2, num_groups):
    B, G, C, H, W = fea1.shape
    cost = torch.pow((fea1 - fea2), 2).sum(2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_struct_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, G, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_difference(refimg_fea[:, :, :, :, i:], targetimg_fea[:, :, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_difference(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def norm_correlation_4D(fea1, fea2): #mq
    B,C,D,H,W = fea1.shape
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    assert cost.shape == (B, 1, D, H, W)
    return cost

def groupwise_correlation_norm_4D(fea1, fea2, num_groups): #mq
    B, C, D, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, D, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, D, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def SpatialTransformer_grid(x, y, disp_range_samples):

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples
    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    return y_warped
        
        
        
def context_upsample(depth_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape
        
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = (depth_unfold*up_weights).sum(1)
        
    return depth


def regression_topk(cost, disparity_samples, k): #cost = (20,48,64,128), disparity_samples = (20,48,64, 128)

    _, ind = cost.sort(1, True) # sort cost along dim = 1, descnding=True, ind = [20,48, 64, 128]
    pool_ind = ind[:, :k] #(20,2,64,128)
    cost = torch.gather(cost, 1, pool_ind) #(20, 2, 64, 128)
    prob = F.softmax(cost, 1) #(20,2,64,128)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)   #(20,2,64,128)
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True) #(20,1,64,128)

    return pred
    

def get_prob_and_disp_topk(cost, disparity_samples, k): #cost = (20,48,64,128), disparity_samples = (20,48,64, 128)
    full_band_prob = F.softmax(cost, 1)

    _, ind = cost.sort(1, True) # sort cost along dim = 1, descnding=True, ind = [20,48, 64, 128]
    pool_ind = ind[:, :k] #(20,2,64,128)
    cost_topk = torch.gather(cost, 1, pool_ind) #(20, 2, 64, 128)
    renormalized_prob_topk = F.softmax(cost_topk, 1) #(20,2,64,128) # renormalized the topk probability
    disparity_candidates_topk = torch.gather(disparity_samples, 1, pool_ind)   #(20,2,64,128)

    return full_band_prob, disparity_candidates_topk, renormalized_prob_topk



