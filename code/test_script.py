'''
This script allows testing multiple datasets and multiple checkpoints 

Shows generalization behaviour and best performing checkpoint

'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from utils import *

import gc
import skimage
import skimage.io

import os
import copy
from collections import OrderedDict
import time
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse


from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012

from datasets import middlebury_loader as mb

from datasets import ETH3D_loader as et
from datasets.readpfm import readPFM
from datasets import readpfm as rp

import cv2

from models import __models__
# from models_acv import __t_models__
# from models_cgi_resnet_full_rec import __models__
# from models_cgi_resnet50 import __models__


# from models_igev.core.igev_stereo import IGEVStereo

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
# parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# parser.add_argument('--datapath', default="/data/KITTI/KITTI_2015/training/", help='data path')
# parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt', default='./pretrained_models/CGI_Stereo/sceneflow.ckpt',help='load the weights from a specific checkpoint')

# parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
# parser.add_argument('--freeze_attention_weights', default=False, type=str,  help='freeze attention weights parameters')

args = parser.parse_args()

def kitti(v,loadckpt):
    
    maxdisp = 192
    datapath12 = "/storage/datasets/kitti/stereo_2012/training/"
    datapath15 = "/storage/datasets/kitti/stereo_2015/training/"

    if v == '2015':
        all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(datapath15)
    else:
        all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(datapath12)

    test_limg = all_limg + test_limg
    test_rimg = all_rimg + test_rimg
    test_ldisp = all_ldisp + test_ldisp

    model = __models__[args.model](maxdisp)
    model = nn.DataParallel(model)
    # model = nn.DataParallel(IGEVStereo(args))

    model.cuda()
    model.eval()

    # model = __models__[args.model](args.maxdisp)
    # model = nn.DataParallel(model)
    # model.cuda()
    # model.eval()

    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

    pred_mae = 0
    pred_op = 0
    for i in trange(len(test_limg)):
        limg = Image.open(test_limg[i]).convert('RGB')
        rimg = Image.open(test_rimg[i]).convert('RGB')

        w, h = limg.size
        m = 32
        wi, hi = (w // m + 1) * m, (h // m + 1) * m
        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        limg_tensor = transform(limg)
        rimg_tensor = transform(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        disp_gt = Image.open(test_ldisp[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
        gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            # pred_disp  = model(limg_tensor, rimg_tensor)[-1]
            pred_disp  = model(limg_tensor, rimg_tensor)
            # print("pred disp: ",len(pred_disp))
            pred_disp = pred_disp[-1]

            pred_disp = pred_disp[:, hi - h:, wi - w:]

        predict_np = pred_disp.squeeze().cpu().numpy()

        op_thresh = 3
        mask = (disp_gt > 0) & (disp_gt < maxdisp)
        # print("Sizes: ",predict_np.size(), mask.size(), disp_gt.size())
        error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

        pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
        pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
        pred_mae += np.mean(pred_error[mask])

        # print("#### >3.0", np.sum((pred_error > op_thresh)) / np.sum(mask))
        # print("#### EPE", np.mean(pred_error[mask]))

    # print("#### EPE", pred_mae / len(test_limg))
    # print("#### >3.0", pred_op / len(test_limg))

    return pred_mae/len(test_limg) , pred_op / len(test_limg)


def mid(loadckpt):
    
    maxdisp = 192
    datapath = "/storage/datasets/kitti/middlebury_2014/MiddEval3/"
    resolution = 'H'


    train_limg, train_rimg, train_gt, test_limg, test_rimg = mb.mb_loader(datapath, res=resolution)

    model = __models__[args.model](maxdisp)
    model = nn.DataParallel(model)
    model.cuda()

    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    os.makedirs('./demo/middlebury/', exist_ok=True)
    
    op = 0
    mae = 0

    for i in trange(len(train_limg)):

        limg_path = train_limg[i]
        rimg_path = train_rimg[i]

        limg = Image.open(limg_path).convert('RGB')
        rimg = Image.open(rimg_path).convert('RGB')

        w, h = limg.size
        wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32

        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_disp  = model(limg_tensor, rimg_tensor)[-1]

            pred_disp = pred_disp[:, hi - h:, wi - w:]

        pred_np = pred_disp.squeeze().cpu().numpy()

        torch.cuda.empty_cache()

        disp_gt, _ = rp.readPFM(train_gt[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
        disp_gt[disp_gt == np.inf] = 0

        occ_mask = Image.open(train_gt[i].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

        mask = (disp_gt <= 0) | (occ_mask != 255) | (disp_gt >= maxdisp)
        # mask = (disp_gt <= 0) | (disp_gt >= args.maxdisp)

        error = np.abs(pred_np - disp_gt)
        error[mask] = 0
        print("#######Bad", limg_path, np.sum(error > 2.0) / (w * h - np.sum(mask)))

        op += np.sum(error > 2.0) / (w * h - np.sum(mask))
        mae += np.sum(error) / (w * h - np.sum(mask))


        #######save

        filename = os.path.join('./demo/middlebury/', limg_path.split('/')[-2]+limg_path.split('/')[-1])
        pred_np_save = np.round(pred_np * 256).astype(np.uint16)        
        cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # print("######Bad 2.0", op / 15 * 100)
    # print("######EPE", mae / 15)
    
    return mae/15, op/15
    

def eth3d(loadckpt):

    maxdisp = 192
    datapath = "/storage/datasets/kitti/eth3d/"

    all_limg, all_rimg, all_disp, all_mask = et.et_loader(datapath)


    model = __models__[args.model](maxdisp)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    os.makedirs('./demo/ETH3D/', exist_ok=True)

    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])


    pred_mae = 0
    pred_op = 0
    for i in trange(len(all_limg)):
        limg = Image.open(all_limg[i]).convert('RGB')
        rimg = Image.open(all_rimg[i]).convert('RGB')

        w, h = limg.size
        wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32
        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        disp_gt, _ = readPFM(all_disp[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
        disp_gt[disp_gt == np.inf] = 0
        gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

        occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

        with torch.no_grad():

            pred_disp  = model(limg_tensor, rimg_tensor)[-1]

            pred_disp = pred_disp[:, hi - h:, wi - w:]

        predict_np = pred_disp.squeeze().cpu().numpy()

        op_thresh = 1
        mask = (disp_gt > 0) & (occ_mask == 255)
        # mask = disp_gt > 0
        error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

        pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
        pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
        pred_mae += np.mean(pred_error[mask])

        ########save

        filename = os.path.join('./demo/ETH3D/', all_limg[i].split('/')[-2]+all_limg[i].split('/')[-1])
        pred_np_save = np.round(predict_np*4 * 256).astype(np.uint16)
        cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])



    # print(pred_mae / len(all_limg))
    # print(pred_op / len(all_limg))
    
    return pred_mae / len(all_limg) , pred_op / len(all_limg)



if __name__ == '__main__':
    print("========================= Testing for Generalization =====================================")
    print(args.loadckpt)
    
    load_5 = args.loadckpt+"checkpoint_000004.ckpt"
    load_10 = args.loadckpt+"checkpoint_000009.ckpt"
    load_15 = args.loadckpt+"checkpoint_000014.ckpt"
    load_16 = args.loadckpt+"checkpoint_000015.ckpt"
    load_17 = args.loadckpt+"checkpoint_000016.ckpt"
    load_18 = args.loadckpt+"checkpoint_000017.ckpt"
    load_19 = args.loadckpt+"checkpoint_000018.ckpt"
    load_20 = args.loadckpt+"checkpoint_000019.ckpt"
    
    loads = [load_5,load_10,load_15,load_16,load_17,load_18,load_19,load_20]
    
    for i in loads:
        print(i)
    
    print("KITTI 2012 TESTING")
    kitti_12 = []
    for i in loads: 
        print("current test ckpt: ",i)
        kitti_12.append([kitti('2012',i)])
        print(kitti_12[-1])

    print("KITTI 2015 TESTING")
    kitti_15 = []
    for i in loads: 
        print("current test ckpt: ",i)
        kitti_15.append([kitti('2015',i)])
        print(kitti_15[-1])

    print("MIDDLEBURY TESTING")
    mid_res = []
    for i in loads: 
        print("current test ckpt: ",i)
        mid_res.append([mid(i)])
        print(mid_res[-1])

    print("ETH3D TESTING")
    eth_res = []
    for i in loads: 
        print("current test ckpt: ",i)
        eth_res.append([eth3d(i)])
        print(eth_res[-1])

    print("SAVING RESULTS")

    with open('test_results.txt', 'w') as f:
        # Define the data to be written
        # data = ['This is the first line', 'This is the second line', 'This is the third line']
        line = "KITTI12_EPE,KITTI12_3.0,KITTI15_EPE,KITTI15_3.0,MID_EPE,MID_2.0,ETH3D_EPE,ETH3D_1.0"
        f.write(line + '\n')

        # Use a for loop to write each line of data to the file
        for i in range(len(loads)):
            
            line = str(kitti_12[i][0][0]) + "," + str(kitti_12[i][0][1]) + "," + str(kitti_15[i][0][0]) + "," + str(kitti_15[i][0][1]) + ","+ str(mid_res[i][0][0]) + "," + str(mid_res[i][0][1]) + "," + str(eth_res[i][0][0]) + "," + str(eth_res[i][0][1])
            f.write(line + '\n')
            # Optionally, print the data as it is written to the file
            # print(line)


