'''

Code for distilling knowledge with masks
The masks are from uncerrtainty maps from ResNET50

'''

from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__

# # 
from models import __models__, model_loss_train, model_loss_test,KD_feat_loss,KD_cvolume_loss,KD_deconv8,KD_deconv4
# from models_cgi_resnet_full import __t_models__, model_loss_train, model_loss_test,KD_feat_loss,KD_cvolume_loss,KD_deconv8,KD_deconv4

# resnet 152 without increased
from models_cgi_resnet_full_rec import __t_models__, model_loss_train, model_loss_test
# from models_cgi_resnet50 import __t_models__, model_loss_train, model_loss_test


# from models_acv import __t_models__, acv_model_loss_train_attn_only, acv_model_loss_train_freeze_attn, acv_model_loss_train, acv_model_loss_test
# from models_cf import __t_models__, model_loss 

from utils import *
from torch.utils.data import DataLoader
import gc
from torchvision.utils import save_image

cudnn.benchmark = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Knowledge Distillation ACVNet to CGI-Stereo')

# Additional Args
parser.add_argument('--t_model', default='CGI_Stereo', help='select a teacher model structure', choices=__t_models__.keys())
parser.add_argument('--t_loadckpt', default='../checkpoints/cgi_resnet_50/second/checkpoint_000018.ckpt', help='load the weights from pretrained teacher')

parser.add_argument('--model', default='CGI_Stereo', help='select a student model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="../data/sceneflow/", help='data path')
parser.add_argument('--trainlist', default='../filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='../filenames/sceneflow_test.txt', help='testing list')

# parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default="/home/xgw/data/KITTI_2015/", help='data path')
# parser.add_argument('--trainlist', default='./filenames/kitti12_15_all.txt', help='training list')
# parser.add_argument('--testlist',default='./filenames/kitti15_all.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=20, help='testing batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="10,14,16,18:2", help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

#STUDENT
# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

#TEACHER RESNET
# model, optimizer
t_model = __t_models__[args.t_model](args.maxdisp)
t_model = nn.DataParallel(t_model)
t_model.cuda()

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(model_dict)

# Loading teacher model
# Have to load pretrained for CFNET
print("loading teacher model {}".format(args.t_loadckpt))
t_state_dict = torch.load(args.t_loadckpt)
print("state dict: ",t_state_dict.keys())
t_model.load_state_dict(t_state_dict['model'])


print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100

    device = 0

    gen1 = nn.Sequential(
            nn.Conv2d(560,560, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(560,560, kernel_size=3, padding=1)).to(device)
    gen2 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(1024,1024, kernel_size=3, padding=1)).to(device)
    gen3 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(2048,2048, kernel_size=3, padding=1)).to(device)

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample,gen1,gen2,gen3,compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


# train one sample
def train_sample(sample, gen1,gen2,gen3, compute_metrics=False):


    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    optimizer.zero_grad()

    disp_ests,s_ll,s_rl,_,_,_,_ = model(imgL, imgR)


    with torch.no_grad():
        # evaluate mode on teacher
        t_model.eval()
        # teacher disp ests
        # t_disp_ests,t_feat,t_cvolume,t_conv4,t_conv8 = t_model(imgL,imgR)
        t_disp_ests,t_ll,t_rl,t_umaps = t_model(imgL,imgR)
        # tumap size 1/4

    # device = t_ll[0].get_device()

    # for i in range(len(t_ll)):
    #     print("Teacher left: ",i,t_ll[i].size(), t_ll[i].get_device())
    # for i in range(len(s_ll)):
    #     print("Student left: ",i,s_ll[i].size(),s_ll[i].get_device())
    # print("Umap: ",t_umaps[0].size(),t_umaps[0].type())



    t_down_umaps = []
    t_down_umaps.append(t_umaps[-1]) #1/4_
    t_down_umaps.append(F.interpolate(t_down_umaps[-1], scale_factor=0.5, mode='bilinear', align_corners=False)) # 1/8
    t_down_umaps.append(F.interpolate(t_down_umaps[-1], scale_factor=0.5, mode='bilinear', align_corners=False)) # 1/16
    t_down_umaps.append(F.interpolate(t_down_umaps[-1], scale_factor=0.5, mode='bilinear', align_corners=False)) # 1/32
    
    # print("umaps len: ",len(t_down_umaps))
    # for i in range(len(t_down_umaps)):
    #     save_image(t_down_umaps[i].float(),'resnetfull_umaps/'+str(i)+'_test_umap.png')

    
    # for i in range(len(t_down_umaps)):
    #     print(" downsampled map index, size ",i,t_down_umaps[i].size())

    '''
    Features from student as s_ll,s_rl
    [1/4,1/8,1/16,1/32]
    Features from teacher as t_ll,t_rl
    [1/2,1/4,1/8,1/16,1/32]
    
    '''

    # left features aligned 
    s_ll[0] = align(s_ll[0],s_ll[0].size()[1],t_ll[0].size()[1])
    s_ll[1] = align(s_ll[1],s_ll[1].size()[1],t_ll[1].size()[1])
    s_ll[2] = align(s_ll[2],s_ll[2].size()[1],t_ll[2].size()[1])
    s_ll[3] = align(s_ll[3],s_ll[3].size()[1],t_ll[3].size()[1])

    # right features aligned 
    s_rl[0] = align(s_rl[0],s_rl[0].size()[1],t_rl[0].size()[1])
    s_rl[1] = align(s_rl[1],s_rl[1].size()[1],t_rl[1].size()[1])
    s_rl[2] = align(s_rl[2],s_rl[2].size()[1],t_rl[2].size()[1])
    s_rl[3] = align(s_rl[3],s_rl[3].size()[1],t_rl[3].size()[1])
    
    # print("Feat align student , teacher: ",s_feat.size(),t_feat.size())
    # print("Volume align student , teacher: ",s_cvolume.size(),t_cvolume.size())
    # print("Conv4 align student , teacher: ",s_conv4.size(),t_conv4.size())
    # print("Conv8 align student , teacher: ",s_conv8.size(),t_conv8.size())

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    masks = [mask, mask_low]
    disp_gts = [disp_gt, disp_gt_low] 

    loss = model_loss_train(disp_ests, disp_gts, masks)
    # print("loss ",loss)

    kd_loss = 0
    feat_loss = 0
    cvolume_loss = 0
    conv4_loss = 0 
    conv8_loss = 0
    
    lambda_feat = 0.001
    lambda_cvolume = 0 
    lambda_conv4 = 0 
    lambda_conv8 = 0 
    
    # using default value
    # change according to usecase
    # classificatio = 0.5
    # semantic = 0.75
    # detection / instance - 0.45
    lambda_mgd = 0.75

    feat_loss = feat_loss + get_dis_loss(s_ll[0], t_ll[0],student_channels=s_ll[0].size()[1], teacher_channels=t_ll[0].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[0], generation = gen1)  
    feat_loss = feat_loss + get_dis_loss(s_ll[1], t_ll[1],student_channels=s_ll[1].size()[1], teacher_channels=t_ll[1].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[1], generation = gen2)  
    feat_loss = feat_loss + get_dis_loss(s_ll[2], t_ll[2],student_channels=s_ll[2].size()[1], teacher_channels=t_ll[2].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[2], generation = gen3)  
    feat_loss = feat_loss + get_dis_loss(s_ll[3], t_ll[3],student_channels=s_ll[3].size()[1], teacher_channels=t_ll[3].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[3], generation = gen3)  

    feat_loss = feat_loss + get_dis_loss(s_rl[0], t_rl[0],student_channels=s_rl[0].size()[1], teacher_channels=t_rl[0].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[0], generation = gen1)  
    feat_loss = feat_loss + get_dis_loss(s_rl[1], t_rl[1],student_channels=s_rl[1].size()[1], teacher_channels=t_rl[1].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[1], generation = gen2)  
    feat_loss = feat_loss + get_dis_loss(s_rl[2], t_rl[2],student_channels=s_rl[2].size()[1], teacher_channels=t_rl[2].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[2], generation = gen3)  
    feat_loss = feat_loss + get_dis_loss(s_rl[3], t_rl[3],student_channels=s_rl[3].size()[1], teacher_channels=t_rl[3].size()[1], lambda_mgd=lambda_mgd, mask = t_down_umaps[3], generation = gen3)  


    # cvolume_loss = KD_cvolume_loss(student=s_cvolume,teacher=t_cvolume) 
    # cvolume_loss = get_dis_loss_3D(preds_S=s_cvolume,preds_T=t_cvolume,student_channels=s_cvolume.size()[1],teacher_channels=t_cvolume.size()[1],lambda_mgd=lambda_mgd) 
    # conv4_loss = KD_deconv4(student=s_conv4,teacher=t_conv4) 
    # conv8_loss = KD_deconv8(student=s_conv8,teacher=t_conv8) 


    kd_loss = kd_loss + lambda_feat * feat_loss + lambda_cvolume * cvolume_loss + \
        lambda_conv4 * conv4_loss + lambda_conv8 * conv8_loss

    # print("feature loss ",feat_loss)
    # print("cvolume loss ",cvolume_loss)
    # print("conv4 loss ",conv4_loss)
    # print("conv8 loss ",conv8_loss)
    # print("loss",loss)

    loss = loss + kd_loss
    # print("loss sum ",loss)
    
    disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
    loss.backward()
    optimizer.step()

    # Add knoledge distillation error here
    return tensor2float(loss), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)

def align(student,student_channels,teacher_channels):
    # Given two tensors of different channel numbers 
    # align the two with a 1x1 kernel
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = student.device

    # print("student and teacher channels",student_channels, teacher_channels)
    # print("Types: ",student.type())
    if student_channels!= teacher_channels and len(student.size())==4:
        m = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0).to(device)
    elif student_channels!= teacher_channels and len(student.size())==5:
        m = nn.Conv3d(student_channels, teacher_channels, kernel_size=(1,1,1), stride=1, padding=0).to(device)
    else:
        m = None
        return student

    return m(student)

def get_dis_loss(preds_S, preds_T,student_channels, teacher_channels, lambda_mgd=0.15, mask=None, generation=None):
    
    N, C, H, W = preds_T.shape

    device = preds_S.device
    
    # print("device: " ,device)

    # generation = nn.Sequential(
    #         nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True), 
    #         nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)).to(device)

    mat = torch.rand((N,1,H,W)).to(device) 
    # print("matrix: " ,mat.size())

    # mask generation
    mat = torch.where(mat < lambda_mgd, 0, 1).to(device)
    # print("matrix: " ,mat.size())

    # threshold for umaps
    # thresh = 0.90

    # if mask is not None:
    #     ma = mask.max()
    #     mi = mask.min()
    #     thr = mi + (ma-mi) * thresh
    #     mat  = torch.where(mask > thr, 0, 1).to(device)

    # save_image(mat.float(),'resnetfull_umaps/0_test_mask.png')
    # # mask aligned student 
    # masked_feat = torch.mul(preds_S, mat)
    # # print("masked_feat: " ,masked_feat.size())
    
    # # Genearate feature from student to be compared with teacher
    # new_feat = generation(masked_feat)
    # # print("New feat: " ,new_feat.size())

    # # calculate distilation loss
    # # check the implementation here for distillation loss
    # # dis_loss = loss_mse(new_feat, preds_T)/N
    # dis_loss = F.mse_loss(new_feat,preds_T)
    dis_loss = F.mse_loss(preds_S,preds_T)
    # print("dis_loss : " ,dis_loss.size())

    return dis_loss

def get_dis_loss_3D(preds_S, preds_T,student_channels, teacher_channels, lambda_mgd=0.15):


    N, C, D, H, W = preds_T.shape

    device = preds_S.device
    
    # print("device: " ,device)

    generation = nn.Sequential(
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=3, padding=1)).to(device)


    mat = torch.rand((N,1,D,H,W)).to(device) 
    # print("matrix: " ,mat.size())

    # mask generation
    mat = torch.where(mat < lambda_mgd, 0, 1).to(device)
    # print("matrix: " ,mat.size())

    # mask aligned student 
    masked_feat = torch.mul(preds_S, mat)
    # print("masked_feat: " ,masked_feat.size())
    
    # Genearate feature from student to be compared with teacher
    new_feat = generation(masked_feat)
    # print("New feat: " ,new_feat.size())

    # calculate distilation loss
    # check the implementation here for distillation loss
    # dis_loss = loss_mse(new_feat, preds_T)/N
    dis_loss = F.mse_loss(new_feat,preds_T)
    # print("dis_loss : " ,dis_loss)

    return dis_loss

if __name__ == '__main__':
    train()
