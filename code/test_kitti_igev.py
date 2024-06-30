import torch
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
# from models import __models__

from models_igev.core.igev_stereo import IGEVStereo

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
# parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/data/KITTI/KITTI_2015/training/", help='data path')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt', default='./pretrained_models/CGI_Stereo/sceneflow.ckpt',help='load the weights from a specific checkpoint')

# igev

parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/igev_sceneflow.pth')
# parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

# Architecure choices
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

# parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
# parser.add_argument('--freeze_attention_weights', default=False, type=str,  help='freeze attention weights parameters')

args = parser.parse_args()


if args.kitti == '2015':
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
else:
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
test_ldisp = all_ldisp + test_ldisp

# model = __models__[args.model](args.maxdisp)
# model = nn.DataParallel(model)
# model.cuda()

t_model = nn.DataParallel(IGEVStereo(args))
t_model.cuda()
t_model.eval()


# state_dict = torch.load(args.loadckpt)
# model.load_state_dict(state_dict['model'])


# Loading for IGEV 
print("loading teacher model {}".format(args.restore_ckpt))
t_state_dict = torch.load(args.restore_ckpt)
print("state dict: ",t_state_dict.keys())
t_model.load_state_dict(t_state_dict,strict=True)

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
        # pred_disp  = model(limg_tensor, rimg_tensor)
        pred_disp = t_model(limg_tensor, rimg_tensor, iters=32, test_mode=True)
        # print("pred disp: ",len(pred_disp))

        pred_disp = pred_disp[-1]
        # pred_disp = pred_disp

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 3
    mask = (disp_gt > 0) & (disp_gt < args.maxdisp)

    # print("Sizes: ",predict_np.size())#, mask.size(), disp_gt.size())
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    # print("#### >3.0", np.sum((pred_error > op_thresh)) / np.sum(mask))
    # print("#### EPE", np.mean(pred_error[mask]))

print("#### EPE", pred_mae / len(test_limg))
print("#### >3.0", pred_op / len(test_limg))
