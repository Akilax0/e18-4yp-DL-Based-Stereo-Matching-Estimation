from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        #return [x4, x8, x16, x32]  # mq
        return [x2, x4, x8, x16, x32] # mq

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)
        self.conv2 = BasicConv(chans[0]*2, chans[0]*2, kernel_size=3, stride=1, padding=1)# check, mq

        self.weight_init()

    def forward(self, featL, featR=None):
        x2, x4, x8, x16, x32 = featL

        y2, y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        x2 = self.deconv4_2(x4, x2)
        y2 = self.deconv4_2(y4, y2)
        x2 = self.conv2(x2)
        y2 = self.conv2(y2)

        #return [x4, x8, x16, x32], [y4, y8, y16, y32]

        return [x2, x4, x8, x16, x32], [y2, y4, y8, y16, y32] #check, mq


class Context_Geometry_Fusion(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(Context_Geometry_Fusion, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.att = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                                 padding=(0,2,2), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                             padding=(0,2,2), stride=1, dilation=1)

        self.weight_init()

    def forward(self, cv, feat):
        '''
        '''
        feat = self.semantic(feat).unsqueeze(2)
        att = self.att(feat+cv)
        cv = torch.sigmoid(att)*feat + cv
        cv = self.agg(cv)
        return cv


class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        return conv


class Multimodal_CGI(nn.Module):
    def __init__(self, k=2, maxdisp=256):
        super(Multimodal_CGI, self).__init__()
        self.maxdisp = maxdisp
        self.k = k   # k-modal
        self.var = 2
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv4 = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc4 = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg4 = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion4 = hourglass_fusion(8)
        self.corr_stem4 = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.conv2 = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc2 = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic2 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg2 = BasicConv(8, 8, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        self.hourglass_fusion2 = hourglass_fusion(8)
        self.corr_stem2 = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

        #CFNet related module
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

    def find_bounds(self, prob):
        # prob [20,48,64,128]
        # max_prob [20,64,128]
        # max_indices [20,64,128]
        # left_bound [20,64,128]
        # left_bound [20,64,128]
        # left_bound[max_indices] [20,64,128,64,128]

        max_probs, max_indices = torch.max(prob, dim=1)
        left_bound = torch.zeros_like(max_indices)
        right_bound = torch.zeros_like(max_indices)


        #Iterate through the probability levels
        for i in range(prob.size(1)):
            print("prob: ",i,prob[0,i,:,:]) 
            
            # This gives dominant peak at each disparity level
            ma_prob = torch.max(prob[0,i,:,:])
            print("max_prob: ",ma_prob)

        print("Debug: max_probs, max_indices, left_bound, right_bound , left_bound[0]", max_probs.size(),max_indices.size(),left_bound.size(),right_bound.size(),left_bound[0].size())

        # #Traverse right to find left bound
        # for i in range(prob.size(2)):
        #     # left_bound[max_indices]=torch.where(prob[torch.arange(prob.size(0)), max_indices, left_bound[max_indices], torch.arange(prob.size(3))] < max_probs, left_bound[max_indices], left_bound[max_indices] - 1)
        #     left_bound[max_indices]=torch.where(prob[torch.arange(prob.size(0)), max_indices, \
        #                                              left_bound[max_indices], torch.arange(prob.size(3))] < max_probs, left_bound[max_indices], left_bound[max_indices] - 1)

        # # Traverse left to find right bound
        # for i in range(prob.size(2)-1, -1, -1):
        #     # right_bound[max_indices] = torch.where(prob[torch.arange(prob.size(0)), max_indices, right_bound[max_indices], torch.arange(prob.size(3))] < max_probs, right_bound[max_indices],right_bound[max_indices] + 1)
        #     right_bound[max_indices] = torch.where(prob[torch.arange(prob.size(0)), max_indices, \
        #                                       right_bound[max_indices], torch.arange(prob.size(3))] < max_probs, right_bound[max_indices],right_bound[max_indices] + 1)
            

        # Iteration happens in the disparity dimension
        #Traverse right to find left bound
        for i in range(prob.size(2)):
            for j in range(prob.size(3)):
                # Iterating through the cells
                
                # max disp for all channels
                max_disp = max_probs[:,i,j]
                max_pos = max_indices[:,i,j]

                print("Max Disp size , pos: ",max_disp.size(),max_pos.size())
                print("spatial position:",i,j)
                print("Max pos: ",max_pos)
                print("Max disp: ",max_disp)

                a = max_disp

                # for k1 in range(max_pos[0]+1,prob.size(1)):
                #     if(a>max_)



            # # left_bound[max_indices]=torch.where(prob[torch.arange(prob.size(0)), max_indices, left_bound[max_indices], torch.arange(prob.size(3))] < max_probs, left_bound[max_indices], left_bound[max_indices] - 1)
            # left_bound[i]=torch.where(prob[torch.arange(prob.size(0)), max_indices, \
            #                                          left_bound[max_indices], torch.arange(prob.size(3))] < max_probs, left_bound[max_indices], left_bound[max_indices] - 1)

        # # Traverse left to find right bound
        # for i in range(prob.size(1)-1, -1, -1):
        #     # right_bound[max_indices] = torch.where(prob[torch.arange(prob.size(0)), max_indices, right_bound[max_indices], torch.arange(prob.size(3))] < max_probs, right_bound[max_indices],right_bound[max_indices] + 1)
        #     right_bound[max_indices] = torch.where(prob[torch.arange(prob.size(0)), max_indices, \
        #                                       right_bound[max_indices], torch.arange(prob.size(3))] < max_probs, right_bound[max_indices],right_bound[max_indices] + 1)
            
        return max_probs, max_indices, left_bound, right_bound



    # III.C for selecting dominant modal
    def select_dominant_modal_disparity(self, prob_dist, disp_samples):
        n, d, h, w = prob_dist.shape
        _, d2, h2, w2 = disp_samples.shape

        assert d==d2 and h==h2 and w==w2
        print("All dimensions ",d,h,w,d2,h2,w2)

        # removing cum prob along disparity dimension
        # What we want to implement is faster left and right bounds
        # cumulative_prob = torch.cumsum(prob_dist, dim=1)
        cumulative_prob = prob_dist

        # Working on calculating the bounds        
        max_probs, max_indices, left_bound, right_bound = self.find_bounds(cumulative_prob)

        print("===============Cumulative prob generated ===================")
        for i in range(w):
            prob_dist[:, max_indices, :, torch.where(torch.arange(n).unsqueeze(1) == max_indices & ((torch.arange(w) < left_bound[max_indices].unsqueeze(1)) | (torch.arange(w) > right_bound.unsqueeze(1))), True, False)]=0.0

        renormalized_prob = F.softmax(prob_dist, dim=1)

        final_pred_disp = torch.sum(renormalized_prob * disp_samples, dim=1, keepdim=True)

        return final_pred_disp


    def forward(self, left, right):
        features_left = self.feature(left) #feature map @ [1/2, 1/4, 1/8, 1/16, 1/32] resolution
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left) # 1/2
        stem_4x = self.stem_4(stem_2x) #1/4
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_2x), 1) #feature map at 1/2
        features_right[0] = torch.cat((features_right[0], stem_2y), 1)
        features_left[1] = torch.cat((features_left[1], stem_4x), 1) #feature map at 1/4
        features_right[1] = torch.cat((features_right[1], stem_4y), 1)


        match_left_4 = self.desc4(self.conv4(features_left[1]))
        match_right_4 = self.desc4(self.conv4(features_right[1]))

        corr_volume_4 = build_norm_correlation_volume(match_left_4, match_right_4, self.maxdisp//4)
        corr_volume_4 = self.corr_stem4(corr_volume_4)
        feat_volume_4 = self.semantic4(features_left[1]).unsqueeze(2)
        volume_4 = self.agg4(feat_volume_4 * corr_volume_4)

        _, *features_left_v2 = features_left
        cost_4 = self.hourglass_fusion4(volume_4, features_left_v2)

        xspx = self.spx_4(features_left[1])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples_4 = torch.arange(0, self.maxdisp//4, dtype=cost_4.dtype, device=cost_4.device) #(48,)
        disp_samples_4 = disp_samples_4.view(1, self.maxdisp//4, 1, 1).repeat(cost_4.shape[0],1,cost_4.shape[3],cost_4.shape[4]) #disp_samples.view(1, self.maxdisp//4, 1, 1)=(1, 48, 1, 1), cost = (20,1, 48, 64, 128), disp_samples = (20, 48, 64, 128)

        # Comment wen using III.C 
        #get prediction at 1/4
        # Squeeze removes the channel dimension 
        # The cost volume is at 1/4
        pred_4 = regression_topk(cost_4.squeeze(1), disp_samples_4, 2) # cost.squeeze(1) = (20, 48, 64, 128)
        pred_4_up = context_upsample(pred_4, spx_pred)

        if self.training:
            return [pred_4_up*4, pred_4.squeeze(1)*4]

        else:
            return [pred_4_up*4]



        # #get distribution and top k candidates at 1/4
        # full_band_prob_4, disp_candidates_topk_4, renormalized_prob_topk_4 = get_prob_and_disp_topk(cost_4.squeeze(1), disp_samples_4, self.k)

        # #idea from Section III.C
        # # Whatever happens happens here
        # pred_dominant_modal_4 = self.select_dominant_modal_disparity(full_band_prob_4, disp_samples_4)

        # pred_dominant_modal_4_up = context_upsample(pred_dominant_modal_4, spx_pred)  # pred_up = [n, h, w]

        # if self.training:
        #     return [pred_dominant_modal_4_up * 4, pred_dominant_modal_4.squeeze(1) * 4]
        # else:
        #     return [pred_dominant_modal_4_up * 4]

