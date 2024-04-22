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
        # right_bound [20,64,128]

        #calculting the max positions (disparity)
        # initializing left and right bound to be of [10,64,128]
        max_probs, max_indices = torch.max(prob, dim=1)
        left_bound = torch.zeros_like(max_indices)
        right_bound = torch.zeros_like(max_indices)
        
        h1 = 2
        w1 = 3

        # For debugging
        h1 = 2
        w1 = 3
        b = 0


        # ========================================= Calculate Right Bound ========================================

        batch_prob = prob[:,:,:,:]

        # Initialize right_bounds to hold all possible right bounds
        right_bounds = torch.ones_like(prob)
        # Compare current disparity level with the next to check bound
        right_bounds[:,:-1,:,:] = batch_prob[:,:-1,:,:]<batch_prob[:,1:,:,:]
        # print("Right Bounds: ",right_bounds.size())

        # Find the positions of TRUE 
        # Inverse the ordering so that the min distance disparity level gets picked
<<<<<<< HEAD
        true_locations_r = torch.nonzero(right_bounds == True,as_tuple=False)
        # print("torch locatuons: ",true_locations_r.size())

        # true_locations_r = torch.flip(true_locations,dims=[0])
=======
        true_locations = torch.nonzero(right_bounds == True)
        true_locations_r = torch.flip(true_locations,dims=[0])
>>>>>>> 57bf64e9f2a634cd1ca1d2edfeec213576416e97

        # Setting right bound to max disp (if nothing gets selected should end in the last disparity level)
        right_bound = right_bound + (self.maxdisp-1)

        # tensors for all selected locations to hold dimensions seperately
        t0 = true_locations_r[:,0]
        t1 = true_locations_r[:,1]
        t2 = true_locations_r[:,2]
        t3 = true_locations_r[:,3]

        # Read only the defined values
        # update_values = right_bound[t0,t2,t3]
        # print("updated values size: ",update_values.size())

        # print("updated_values: ",update_values,t0,t1,t2,t3)

        # Get the difference from the max indice and check if it is on the right side of it
        # if not remove the values 
        differences = t1-max_indices[t0,t2,t3]
        # for i in range(len(t0)):
        #     if t2[i]==h1 and t3[i]==w1:
        #         print("Difference, t1, max disp: ",differences[i],t1[i],max_indices[t0[i],t2[i],t3[i]])
        mask = differences >= 0
        # Contains true_loication_r positions which on the right side of the max indice
        update_values = true_locations_r[mask]
        # print("update_value: ",update_values.size())

        # update_values = update_values[mask]
        t0 = t0[mask]
        t1 = t1[mask]
        t2 = t2[mask]
        t3 = t3[mask]

        # Testing too get the minimum 
        pos_2_3 = update_values[:,[0,2,3]]
        # print("pos_2_3 size: ",pos_2_3.size())
        
        # Find unique positions 
        uniq_pos, indices = torch.unique(pos_2_3, dim=0, return_inverse=True)
        # print("uniq pos, indices: ",uniq_pos.size(),indices.size(), indices.get_device())

        # mask_2_3 = torch.eq(pos_2_3)
        # min values tensor
        # min_values = torch.zeros(uniq_pos.size(0))

        # print("=============================update value right============================")
        # for i in range(len(update_values)):
        #     if update_values[i][2]==h1 and update_values[i][3]==w1:
        #         print("true location index , location : ",i,update_values[i])

        for i, pos in enumerate(uniq_pos):
            # MAsk to select sam 2nd and 3rd positions
            mask = indices == i 
            # mask size 404295 <= all avaliable positions (same size as indices) 

            # Apply mask to extract tensors with same  0th ,2nd and 3rd positions
            filtered_val = update_values[mask]
            min_value = torch.min(filtered_val[:,1])

            # min_values[i] = min_value
            update_values[mask] = min_value
            
        # # Have to get masks for all uniq posisitons 
        # #masks size : [uniq_pos , indices]
        # masks = indices.unsqueeze(0).cuda() == torch.arange(uniq_pos.size(0)).unsqueeze(1).cuda()
        # print("masks size: ",masks.size())
        # # Now have to apply this mask to all the update values size : [indices, 4]
        # # Create a tensor sized [uniq_pos,indices, 4] to apply the mask
        # exp_update = torch.ones(uniq_pos.size()[0]).unsqueeze(1).unsqueeze(1).cuda()
        # # exp_update = update_values.unsqueeze(0).tile(indices.size()[0],1,1)
        
        # # exp_update = exp_update * update_values
        # print("exp_update size : ",exp_update.size())
        # exp_update = exp_update.expand(-1, update_values.size()[0],update_values.size()[1])
        # print("exp_update size : ",exp_update.size())
        # exp_update = exp_update * update_values
        # print("exp_update size : ",exp_update.size())

        # # print("mask non zero: ",torch.nonzero(mask))
        # # print("Sizes mask, filtered_val, min_value : ", mask.size(),filtered_val.size(),min_value.size() )

        # print("updated value: ",update_values.size())
        # print("indices and uniq pos: ",indices.unsqueeze(0).size(), torch.arange(uniq_pos.size(0)).unsqueeze(1).size())
        # # print("Mask size: ",masks.any(dim=0).size())

        # filtered_val = update_values[masks]
        # print("filtered val size: ",filtered_val.size())
        # update_values[masks,1] = torch.min(filtered_val[:,1],dim=0)[0]


        # min_values = torch.zeros(uniq_pos.size(0))
        # min_value_indices = torch.min(update_values[:,1].unsqueeze(1)*(indices.unsqueeze(0) == torch.arange(uniq_pos.size(0)).unsqueeze(1)),dim=1)[1]
        # update_values[:,1] = min_values[min_value_indices]



        # print("min values: ",min_values.size())
        # for i in range(len(uniq_pos)):
        #     if uniq_pos[i][0] == h1 and uniq_pos[i][1]==w1:
        #         print("min value at h1,w1 : ",h1,w1,min_values[i])

        # print("updated_values: ",update_values,t0,t1,t2,t3)

        # # Get the minimum distanced disparity level on the right side
        # update_values = torch.minimum(t1,update_values[:,1]) #+ 1) * ((t1-max_indices[t0,t2,t3]>=0))
        # # print("updated value: ",update_values)

        # # If  masked get the right edge (max disp)
        # update_values = torch.where(update_values > 0, update_values, right_bound[t0, t2, t3])
        # # print("updated value: ",update_values)

        # update the right bound
        right_bound[t0,t2,t3] = update_values[:,1]
        # print("right bound: ",right_bound)

        # =============================================================== Calculating Left Bound ============================================================
        
        # Initialize and comapre with previous disparity if left bound
        left_bounds = torch.ones_like(prob)
        left_bounds[:,1:,:,:] = batch_prob[:,1:,:,:]<batch_prob[:,:-1,:,:]
        # print("Left Bounds: ",left_bounds)

        # Find the positions of TRUE 
        true_locations_l = torch.nonzero(left_bounds == True)
        # print("True Locations (Right): ",len(true_locations))

        # # print("Left bound: ",left_bound)
        t0 = true_locations_l[:,0]
        t1 = true_locations_l[:,1]
        t2 = true_locations_l[:,2]
        t3 = true_locations_l[:,3]

        # update_values = left_bound[t0,t2,t3]
        # print("updated_values: ",update_values,t0,t1,t2,t3)

        # Remove the locations that are on the right side of max disparity
        differences = max_indices[t0,t2,t3] - t1
        mask = differences >= 0
        update_values = true_locations_l[mask]
        t0 = t0[mask]
        t1 = t1[mask]
        t2 = t2[mask]
        t3 = t3[mask]
        # print("updated_values: ",update_values,t0,t1,t2,t3)

        # Testing too get the minimum 
        pos_2_3 = update_values[:,[0,2,3]]
        # print("pos_2_3 size: ",pos_2_3.size())
        
        # Find unique positions 
        uniq_pos, indices = torch.unique(pos_2_3, dim=0, return_inverse=True)
        # print("uniq pos, indices: ",uniq_pos.size(),indices.size())

        for i, pos in enumerate(uniq_pos):
            # MAsk to select sam 2nd and 3rd positions
            mask = indices == i

            # Apply mask to extract tensors with 2nd and 3rd positions
            filtered_val = update_values[mask]
            max_value = torch.max(filtered_val[:,1])

            # min_values[i] = min_value
            update_values[mask] = max_value


        # update_values = torch.maximum(t1,update_values)
        # #* (( max_indices[t0,t2,t3] - t1)>=0) 
        # update_values = torch.where(update_values > 0, update_values, left_bound[t0, t2, t3])

        # Update left bound 
        left_bound[t0,t2,t3] = update_values[:,1]

        # print("left bound: ",left_bound)

        print("==================right======================")
        for i in range(len(true_locations_r)):
            if true_locations_r[i][2]==h1 and true_locations_r[i][3]==w1:
                print("true location index , location : ",i,true_locations_r[i])
        print("==================left======================")
        for i in range(len(true_locations_l)):
            if true_locations_l[i][2]==h1 and true_locations_l[i][3]==w1:
                print("true location index , location : ",i,true_locations_l[i])
        print("prob: ",prob[0,:,h1,w1])
        print("right bounds: ",right_bounds[0,:,h1,w1])
        print("left bounds: ",left_bounds[0,:,h1,w1])
        print("right bound: ",right_bound[0,h1,w1])
        print("left bound: ",left_bound[0,h1,w1])
        print("max indices : ",max_indices[0,h1,w1])
        print("max probs: ",max_probs[0,h1,w1])

        # print("=============================right============================")
        # for i in range(len(true_locations_r)):
        #     if true_locations_r[i][2]==h1 and true_locations_r[i][3]==w1:
        #         print("true location index , location : ",i,true_locations_r[i])
        # print("=============================left============================")
        # for i in range(len(true_locations_l)):
        #     if true_locations_l[i][2]==h1 and true_locations_l[i][3]==w1:
        #         print("true location index , location : ",i,true_locations_l[i])
        # print("prob: ",prob[b,:,h1,w1])
        # print("right bounds: ",right_bounds[b,:,h1,w1])
        # print("left bounds: ",left_bounds[b,:,h1,w1])
        # print("right bound: ",right_bound[b,h1,w1])
        # print("left bound: ",left_bound[b,h1,w1])
        # print("max indices : ",max_indices[b,h1,w1])
        # print("max probs: ",max_probs[b,h1,w1])

#==========================================================================================================================

# NAIVE APPROACH

        # # prob [20,48,64,128]
        # # max_prob [20,64,128]
        # # max_indices [20,64,128]
        # # left_bound [20,64,128]
        # # right_bound [20,64,128]
        # for c in range(prob.size(0)):
        #     for i in range(prob.size(2)):
        #         for j in range(prob.size(3)):

        #             # ma_d = max_probs[c,i,j]
        #             ma_x = max_indices[c,i,j]
        #             right = ma_x

        #             for d in range(ma_x+1,prob.size(1)):
        #                 if(prob[c,d,i,j]<=prob[c,d-1,i,j]):
        #                     right = d
        #                 else:
        #                     break;

        #             right_bound[c,i,j] = right
            
        #             left = ma_x

        #             for d in range(ma_x-1,0,-1):
        #                 if(prob[c,d,i,j]<=prob[c,d+1,i,j]):
        #                     left = d
        #                 else:
        #                     break;

        #             left_bound[c,i,j] = left

        #             # print("Bounds for pixel i,j, left, right, max: ",i,j,left,right,ma_x) 


#==========================================================================================================================
            
        return max_probs, max_indices, left_bound, right_bound



    # III.C for selecting dominant modal
    def select_dominant_modal_disparity(self, prob_dist, disp_samples):
        n, d, h, w = prob_dist.shape
        _, d2, h2, w2 = disp_samples.shape

        assert d==d2 and h==h2 and w==w2
        # print("All dimensions ",d,h,w,d2,h2,w2)

        # removing cum prob along disparity dimension
        # What we want to implement is faster left and right bounds
        # cumulative_prob = torch.cumsum(prob_dist, dim=1)
        
        # This has to be changed to getting cumulativve neighbourhood
        # cumulative_prob = prob_dist

        shift_right = torch.zeros_like(prob_dist)        
        shift_left = torch.zeros_like(prob_dist)        


        # Adding the neighbourhood of 1 have to get a param here to adjust
        shift_right[:,1:,:,:] = prob_dist[:,1:,:,:]
        shift_left[:,:-1,:,:] = prob_dist[:,:-1,:,:]
        cumulative_prob = shift_left + shift_right + prob_dist
        
        # print("cum prob:", cumulative_prob.size())

        # Working on calculating the bounds        
        max_probs, max_indices, left_bound, right_bound = self.find_bounds(cumulative_prob)


        n, d, h, w = prob_dist.size()

        # Expand left_bound and right_bound to match the dimensions of prob_dist
        left_bound_expanded = left_bound.unsqueeze(1).expand(n, d, h, w)
        right_bound_expanded = right_bound.unsqueeze(1).expand(n, d, h, w)

        # Create masks for values outside the bounds
        left_mask = prob_dist < left_bound_expanded
        right_mask = prob_dist > right_bound_expanded
        outside_bounds_mask = left_mask | right_mask

        # This inverts the binary values
        outside_bounds_mask = ~outside_bounds_mask 
        # print("outside_bounds mask: ",outside_bounds_mask.size())
        # Zero out values outside the bounds
        # prob_dist[outside_bounds_mask] = 0
        prob_dist = prob_dist * outside_bounds_mask

        renormalized_prob = F.softmax(prob_dist, dim=1)

        final_pred_disp = torch.sum(renormalized_prob * disp_samples, dim=1, keepdim=True)
        
        # print("final pred disp: ",final_pred_disp.size())

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

        # # Comment wen using III.C 
        # #get prediction at 1/4
        # # Squeeze removes the channel dimension 
        # # The cost volume is at 1/4
        # pred_4 = regression_topk(cost_4.squeeze(1), disp_samples_4, 2) # cost.squeeze(1) = (20, 48, 64, 128)
        # pred_4_up = context_upsample(pred_4, spx_pred)

        # if self.training:
        #     return [pred_4_up*4, pred_4.squeeze(1)*4]

        # else:
        #     return [pred_4_up*4]



        #get distribution and top k candidates at 1/4
        full_band_prob_4, disp_candidates_topk_4, renormalized_prob_topk_4 = get_prob_and_disp_topk(cost_4.squeeze(1), disp_samples_4, self.k)

        #idea from Section III.C
        # Whatever happens happens here
        pred_dominant_modal_4 = self.select_dominant_modal_disparity(full_band_prob_4, disp_samples_4)

        pred_dominant_modal_4_up = context_upsample(pred_dominant_modal_4, spx_pred)  # pred_up = [n, h, w]

        if self.training:
            return [pred_dominant_modal_4_up * 4, pred_dominant_modal_4.squeeze(1) * 4]
        else:
            return [pred_dominant_modal_4_up * 4]

