import warnings

import torch
import torch.nn.functional as F
from .multimodal_losses import *


def isNaN(x):
    return x != x


class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume
        Args:
            maxDisp, (int): the maximum of disparity
            gtDisp, (torch.Tensor): in (..., Height, Width) layout
            start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index

        Outputs:
            probability, (torch.Tensor): in [BatchSize, maxDisp, Height, Width] layout


    """
    def __init__(self, maxDisp, gtDisp, start_disp=0, dilation=1):

        if not isinstance(maxDisp, int):
            raise TypeError('int is expected, got {}'.format(type(maxDisp)))

        if not torch.is_tensor(gtDisp):
            raise TypeError('tensor is expected, got {}'.format(type(gtDisp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))

        if gtDisp.dim() == 2:  # single image H x W
            gtDisp = gtDisp.view(1, 1, gtDisp.size(0), gtDisp.size(1))

        if gtDisp.dim() == 3:  # multi image B x H x W
            gtDisp = gtDisp.view(gtDisp.size(0), 1, gtDisp.size(1), gtDisp.size(2))

        if gtDisp.dim() == 4:
            if gtDisp.size(1) == 1:  # mult image B x 1 x H x W
                gtDisp = gtDisp
            else:
                raise ValueError('2nd dimension size should be 1, got {}'.format(gtDisp.size(1)))

        self.gtDisp = gtDisp
        self.maxDisp = maxDisp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + maxDisp - 1
        self.disp_sample_number = (maxDisp + dilation -1) // dilation
        self.eps = 1e-40

    def getProb(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        index = torch.linspace(self.start_disp, self.end_disp, self.disp_sample_number)
        index = index.to(self.gtDisp.device)

        # [BatchSize, maxDisp, Height, Width]
        self.index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()

        # the gtDisp must be (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        probability = self.calProb()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        probability = probability * mask + self.eps


        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {}, max: {}'.format(probability.min(), probability.max()))
            print('Disparity Ground Truth after mask out ==> min: {}, max: {}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'probability contains NaN!")

        return probability

    def kick_invalid_half(self):
        distance = self.gtDisp - self.index
        invalid_index = distance < 0
        # after softmax, the valid index with value 1e6 will approximately get 0
        distance[invalid_index] = 1e6
        return distance

    def calProb(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(LaplaceDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        scaled_distance = ((-torch.abs(self.index - self.gtDisp)) / self.variance)
        # print("scaled distance size: ",scaled_distance.size())
        probability = F.softmax(scaled_distance, dim=1)

        return probability


class GaussianDisp2Prob(Disp2Prob):
    # variance is the variance of the Gaussian distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(GaussianDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt})^2 / b), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        distance = (torch.abs(self.index - self.gtDisp))
        scaled_distance = (- distance.pow(2.0) / self.variance)
        probability = F.softmax(scaled_distance, dim=1)

        return probability


class OneHotDisp2Prob(Disp2Prob):
    # variance is the variance of the OneHot distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(OneHotDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def getProb(self):

        # |d - d{gt}| < variance, [BatchSize, maxDisp, Height, Width]
        probability = torch.lt(torch.abs(self.index - self.gtDisp), self.variance).type_as(self.gtDisp)

        return probability

def fetch_neighborhood_info(gtDisp, m, n, thres, alpha):
    # The pseudo code is for this section
    
    assert len(gtDisp.shape) == 4

    print("gtDisp Nan? ",torch.isnan(gtDisp).any())
    b, c, h, w = gtDisp.shape
    
    print("GT Disparity size: ", gtDisp.size())

    # unfolded_gtDisp = F.unfold(gtDisp, kernel_size=(m,n), padding=(m//2, n//2))
    unfolded_gtDisp = F.pad(gtDisp,pad=(m//2,m//2,n//2,n//2), mode='constant',value=0) 
    print("unfolded (padded) disparity: ",unfolded_gtDisp.size())
    

    
    # neighborhood = unfolded_gtDisp.view(b , m*n, h, w )
    # neighborhood = unfolded_gtDisp.view(b , 1, h+m, w+n )
    neighborhood = unfolded_gtDisp
    print("neighbourhood: ",neighborhood.size())

    # p1_p2_cluster = torch.abs(neighborhood - gtDisp ) > thres # 0: P1 cluster , 1: P0 cluster
    p1_p2_cluster = # from pseudo code 
    print("p1_p2_cluster: ",p1_p2_cluster.size())

    p2_count = torch.sum(p1_p2_cluster, dim=1, keepdim=True)

    total_count = torch.ones((b,1,h,w)) * (m*n)
    # Get to same device
    total_count = total_count.to(p2_count.get_device())

    p1_count = total_count - p2_count
    
    print("p2_count",p2_count.eq(0).any())
    
    # p2_ount contains 0 resulting in NaNs 
    # what can we do here?
    mu2 = torch.sum(p1_p2_cluster * neighborhood, dim=1, keepdim=True) / p2_count # pay attention divided by 0

    w = alpha + (p1_count -1) * (1-alpha) * (m*n-1)

    print("w ",torch.isnan(w).any())
    return mu2, w


def generate_md_gt_distribution(gt, m, n, th1, th2, maxDisp, alpha=0.8):
    # Getting batch size and channel of gt
    b,h,w = gt.size()

    # kernel = torch.ones(1, 1, m, n).to(gt.device)
    kernel = torch.ones(b,1, m, n).to(gt.device)
    
    gt = torch.unsqueeze(gt,dim=1)

    print("kernel gt ",kernel.size(),gt.size())
    mean_gt = F.conv2d(gt, kernel, padding=(m // 2, n // 2))
    mean_gt /= m * n

    # edge - 1  non edge - 0
    edge_mask = torch.abs(gt - mean_gt) > th1

    print("gt ",torch.isnan(gt).any())
    dist1 = LaplaceDisp2Prob(maxDisp, gt, variance=1)
    non_edge_prob_dist = dist1.getProb()

    # mu2 - means of the disparities P2?
    # mu2 gt disp
    mu2, w = fetch_neighborhood_info(gt, m,n, th2, alpha)
    
    print("mu2 ",torch.isnan(mu2).any())
    dist2 = LaplaceDisp2Prob(maxDisp, mu2, variance=1)

    print("Good until here") 
    print("dist2 ",dist2)
    # Equation on paper
    edge_prob_dist = w*non_edge_prob_dist + (1-w)*dist2.getProb()

    md_gt_dist = (~edge_mask) * non_edge_prob_dist + edge_mask * edge_prob_dist

    return md_gt_dist


