import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms
import skimage.io
import os
import copy
from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import argparse
import cv2

from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012
from models import __models__

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/data/KITTI/KITTI_2015/training/", help='data path')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt', default='./pretrained_models/CGI_Stereo/sceneflow.ckpt',help='load the weights from a specific checkpoint')

args = parser.parse_args()

if args.kitti == '2015':
    # all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
    all_limg, all_rimg, test_limg, test_rimg = kt2015.kt2015_bench_loader(args.datapath)
else:
    # all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)
    all_limg, all_rimg, test_limg, test_rimg = kt2012.kt2012_bench_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
# test_ldisp = all_ldisp + test_ldisp

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

pred_mae = 0
pred_op = 0

for i in trange(len(test_limg)):
    
    # print(test_limg[i][-13:])
    suffix = test_limg[i][-13:]
    
    limg = Image.open(test_limg[i]).convert('RGB')
    rimg = Image.open(test_rimg[i]).convert('RGB')

    w, h = limg.size
    # print(limg.size)
    m = 32
    wi, hi = (w // m + 1) * m, (h // m + 1) * m
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))
    # print(limg.size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # transform = transforms.ToTensor()

    limg_tensor = transform(limg)
    rimg_tensor = transform(rimg)
    
    # print(limg_tensor.shape)
   
    # limg_tensor = limg_tensor.cuda()
    # rimg_tensor = rimg_tensor.cuda()
    
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()
 
    # disp_gt = Image.open(test_ldisp[i])
    # disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
    # gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    st = time.time()

    with torch.no_grad():
        # pred_disp  = model(limg_tensor, rimg_tensor)[-1]
        pred_disp  = model(limg_tensor, rimg_tensor)
        # print("pred disp: ",len(pred_disp))
        pred_disp = pred_disp[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]


    maxdisp = 192
    
    predict_np = pred_disp.squeeze().cpu().numpy()
    # mask = (predict_np > 0) & (predict_np < maxdisp)
    # print(predict_np)
    # predict_np = predict_np * mask.astype(np.float32)
    # print(predict_np)
    
    predict_np = (predict_np*256).astype(np.uint16)
    # print("max min: ",np.max(predict_np),np.min(predict_np))
    # print(predict_np.type())
    
    
    if args.kitti == '2015':
        # print(predict_np)
        # image = Image.fromarray(predict_np)
        # image.save('kitti_2015_results/'+suffix)

        # print(predict_np.size())
        # plt.imsave('kitti_2015_results/'+suffix,predict_np,cmap="gray",format='png',vmin=0,vmax=65535)
        # matplotlib.image.imsave('kitti_2015_results/'+suffix,predict_np)
        cv2.imwrite('kitti_2015_results/'+suffix,predict_np)

    else:    
        image = Image.fromarray(predict_np)
        image.save('kitti_2012_results/'+suffix)

    # op_thresh = 3
    # mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
    # # print("Sizes: ",predict_np.size(), mask.size(), disp_gt.size())
    # error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    # pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    # pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    # pred_mae += np.mean(pred_error[mask])
    
    # time_taken = time.time() - st

    # print("#### >3.0", np.sum((pred_error > op_thresh)) / np.sum(mask))
    # print("#### EPE", np.mean(pred_error[mask]))

# print("#### EPE", pred_mae / len(test_limg))
# print("#### >3.0", pred_op / len(test_limg))
# print("TIME: ",time_taken)