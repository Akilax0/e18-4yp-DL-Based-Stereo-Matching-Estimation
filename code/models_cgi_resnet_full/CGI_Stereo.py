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

from .ResNet import *

from .mgd import *

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


# class Feature(SubModule):
#     def __init__(self):
#         super(Feature, self).__init__()
#         pretrained =  True
#         model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
#         layers = [1,2,3,5,6]
#         chans = [16, 24, 32, 96, 160]
#         self.conv_stem = model.conv_stem
#         self.bn1 = model.bn1
#         self.act1 = model.act1

#         self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
#         self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
#         self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
#         self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
#         self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

#         self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

#     def forward(self, x):
#         x = self.act1(self.bn1(self.conv_stem(x)))
#         x2 = self.block0(x)
#         x4 = self.block1(x2)
#         # return x4,x4,x4,x4
#         x8 = self.block2(x4)
#         x16 = self.block3(x8)
#         x32 = self.block4(x16)
#         return [x4, x8, x16, x32]


class Feature(SubModule):
    # feaure to be replaced by resnet152
    def __init__(self):
        super(Feature, self).__init__()
        # model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        model = timm.create_model('resnet152', pretrained=True)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
    def forward(self, x):
        # print("x: ",x.size())
        
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x4 = self.layer1(x)
        x8 = self.layer2(x4)
        x16 = self.layer3(x8)
        x32 = self.layer4(x16)
        # print("x: ",x.size())
        # print("x4: ",x4.size())
        # print("x8: ",x8.size())
        # print("x16: ",x16.size())
        # print("x32: ",x32.size())

        return [x4,x8,x16,x32] 



class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        # chans = [16, 24, 32, 96, 160]
        chans = [64, 256, 512, 1024, 2048]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR
        
        # print("x4: ",x4.size())
        # print("x8: ",x8.size())
        # print("x16: ",x16.size())
        # print("x32: ",x32.size())
        
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        # print("x4: ",x4.size())
        # print("x8: ",x8.size())
        # print("x16: ",x16.size())
        # print("x32: ",x32.size())
        return [x4, x8, x16,x32], [y4, y8, y16,y32]


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


        # self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 2048)
        # self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 2048)
        # self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 1024)

    def forward(self, x, imgs):
        # print("x: ",x.size())
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        # print("done")
        # print("imgs_3: ", imgs[3].size())
        # print("imgs_2: ", imgs[2].size())
        # print("imgs_1: ", imgs[1].size())
        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        
        # to be used for distillation
        # for starter we shall take conv1 and conv
        # later check suitable position before or after CGF

        conv1 = self.CGF_8(conv1, imgs[1])
        # print("conv1 after CGF ",conv1.size())

        conv = self.conv1_up(conv1)
        # print("conv last",conv.size())

        return conv,conv1


class CGI_Stereo(nn.Module):
    def __init__(self, maxdisp):
        super(CGI_Stereo, self).__init__()
        self.maxdisp = maxdisp 
        self.feature = Feature()
        # self.feature_ext = Feature_Ext()
        self.feature_up = FeatUp()
        
        # chans = [16, 24, 32, 96, 160]
        chans = [64, 256, 512, 1024, 2048]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        # Implemented to match the channels no
        self.stem_2_ext = nn.Sequential(
            BasicConv(32, 192, kernel_size=1, stride=1, padding=0))
        
        
        # self.stem_4 = nn.Sequential(
        #     BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(48, 48, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(48), nn.ReLU()
        #     )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
            )

        # self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*192, 9, kernel_size=4, stride=2, padding=1),)
        

        # self.spx_2 = Conv2x(32, 32, True)
        self.spx_2 = Conv2x(192, 192, True)
        # self.spx_4 = nn.Sequential(
        #     BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(32, 32, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(32), nn.ReLU()
        #     )
        self.spx_4 = nn.Sequential(
            BasicConv(576, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU()
            )

        # self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.conv = BasicConv(576, 288, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(288, 288, kernel_size=1, padding=0, stride=1)
        
        
        # self.semantic = nn.Sequential(
        #     BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))

        self.semantic = nn.Sequential(
            BasicConv(576, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 48, kernel_size=1, padding=0, stride=1, bias=False))
        
        
        # self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.agg = BasicConv(48, 48, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        # self.hourglass_fusion = hourglass_fusion(8)
        self.hourglass_fusion = hourglass_fusion(48)
        # self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem = BasicConv(1, 48, is_3d=True, kernel_size=3, stride=1, padding=1)

    def forward(self, left, right):

        features_left = self.feature(left)
        features_right = self.feature(right)

        # print("left image: ",left.size()) 
        # print("feature left: ",features_left.size())

        features_left, features_right = self.feature_up(features_left, features_right)

        # Upscaled 2,4,8,16
        ll = features_left
        rl = features_right
        
        # print("left , right features: ",len(ll))
        # print("feature_left: ",features_left[2].size())
        
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

 
        # print("left , right : ",left.size(),right.size())
        # print("stem_2x , stem_4x size: ",stem_2x.size(), stem_4x.size())

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)
        # print("feature map 1/4 : ",features_left[0].size())


        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        # print("match _left : ",match_left.size())

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        # print("corr_volume : ",corr_volume.size())
        corr_volume = self.corr_stem(corr_volume)
        # print("corr_volume : ",corr_volume.size())
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)

        # print("corr_volume: ",corr_volume.size())
        # print("feat volume: ",feat_volume.size())
        volume = self.agg(feat_volume * corr_volume)
        # print("volume: ",volume.size())
        cost,conv8= self.hourglass_fusion(volume, features_left)
        # print("volume: ",volume.size())

        xspx = self.spx_4(features_left[0])
        # print("xspx:",xspx.size())
        # print("stem2x: ",stem_2x.size())
        
        # Matching channel no
        stem_2x = self.stem_2_ext(stem_2x)

        xspx = self.spx_2(xspx, stem_2x)
        # print("xspx:",xspx.size())
        spx_pred = self.spx(xspx)
        # print("spx_pred: ",spx_pred.size())
        spx_pred = F.softmax(spx_pred, 1)
        # print("spx_pred: ",spx_pred.size())

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        # print("disp samples:",disp_samples.size()) 
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        # print("disp samples:",disp_samples.size()) 
        pred,prob = regression_topk(cost.squeeze(1), disp_samples, 2, self.maxdisp//4)
        # print("pred: ",pred.size())
        pred_up = context_upsample(pred, spx_pred)
        # print("pred up: ",pred_up.size())

        # # Calculting umap
        pred2_cur = pred.detach()
        # # Please check this i think this is wrong
        # pred2_umap = disparity_variance_confidence(prob, self.maxdisp//4, pred2_cur)
        pred2_umap = disparity_variance(prob, self.maxdisp//4, pred2_cur)

        if self.training:
            # outputting left and right features , but not used for training
            return [pred_up*4, pred.squeeze(1)*4]
        else:
            return [pred_up*4],ll,rl,[pred2_umap]
