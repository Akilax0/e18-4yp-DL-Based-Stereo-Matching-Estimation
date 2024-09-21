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
import argparse

from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012
from models import __models__

# from models_acv import __t_models__
# from models_cgi_resnet_full import __models__

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/data/KITTI/KITTI_2015/training/", help='data path')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt', default='./pretrained_models/CGI_Stereo/sceneflow.ckpt',help='load the weights from a specific checkpoint')

# parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
# parser.add_argument('--freeze_attention_weights', default=False, type=str,  help='freeze attention weights parameters')

args = parser.parse_args()

def compute_outliers(gt_disp, pred_disp, mask=None, threshold=3.0):
    """Compute disparity outliers."""
    valid_mask = gt_disp > 0
    if mask is not None:
        valid_mask = valid_mask & mask
       
    device = pred_disp.get_device()
        
    valid_mask = torch.from_numpy(valid_mask).to(device)
    gt_disp = torch.from_numpy(gt_disp).to(device)
    pred_disp = pred_disp.squeeze()
    # pred_disp = torch.from_numpy(pred_disp)
   
    # valid_mask = valid_mask.unsqueeze(0) 
    
    # valid_mask = np.expand_dims(valid_mask, axis=0)
    # y = np.expand_dims(x, axis=0)
    # print("mask: ",valid_mask.size())
    
    error = torch.abs(gt_disp[valid_mask] - pred_disp[valid_mask])
    outliers = error > threshold
    outliers = outliers.detach().cpu().numpy()
    outlier_percentage = np.mean(outliers) * 100
    return outlier_percentage


if args.kitti == '2015':
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
else:
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
test_ldisp = all_ldisp + test_ldisp

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

# model = __models__[args.model](args.maxdisp)
# model = nn.DataParallel(model)
# model.cuda()
# model.eval()

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


D1_all = 0

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


    st = time.time()

    with torch.no_grad():
        # pred_disp  = model(limg_tensor, rimg_tensor)[-1]
        pred_disp  = model(limg_tensor, rimg_tensor)
        # print("pred disp: ",len(pred_disp))
        pred_disp = pred_disp[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 3
    mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
    # print("Sizes: ",predict_np.size(), mask.size(), disp_gt.size())
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])
  
  
   
    # seg_bg = load_segmentation_mask(seg_bg_path) == 255
    # seg_fg = load_segmentation_mask(seg_fg_path) == 255
    D1_all += compute_outliers(disp_gt, pred_disp)

    time_taken = time.time() - st

    # print("#### >3.0", np.sum((pred_error > op_thresh)) / np.sum(mask))
    # print("#### EPE", np.mean(pred_error[mask]))
print("D1: ",D1_all/len(test_limg))
print("#### EPE", pred_mae / len(test_limg))
print("#### >3.0", pred_op / len(test_limg))
print("TIME: ",time_taken)