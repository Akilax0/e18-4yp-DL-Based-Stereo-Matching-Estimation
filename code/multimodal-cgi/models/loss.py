import torch.nn.functional as F
import torch
from .disp2prob import generate_md_gt_distribution


# cross entropy based distribution loss
def ce_based_distribution_loss(prob, disp_gt, img_mask, max_disp):  # cost = (20,48,64,128)

    #pred_prob_distribution = F.log_softmax(cost, 1)  # (20,2,64,128)
    log_pred_prob_distribution = torch.log(prob) # (20,1,64,128)

    #transfer gt disparity map to probability distribution map
    m = 1
    n = 9 # m x n neighborhood to determine whether the centering pixel is edge or not
    th1 = 1 # threshold to determine whether it is edge or not
    th2 = 1 # threshold to determine p1-p2 clustering
    alpha = 0.8
    
    # Just debugging to check if masking works 
    # print("Mask: ",img_mask)
    # print("log pred prob : ",log_pred_prob_distribution)
    # masked_log_pred_prob_dist = log_pred_prob_distribution[img_mask]
    # print("Masked log pred prob: ",masked_log_pred_prob_dist.size())
    # masked_log_pred_prob_dist = log_pred_prob_distribution*img_mask
    # print("Masked log pred prob multi: ",masked_log_pred_prob_dist)

    #idea from III.B
    #making edges more edgier
    gt_prob_dist = generate_md_gt_distribution(disp_gt, m, n, th1, th2, max_disp, alpha)

    # print("Image mask: ",img_mask.size())
    # print("gt_prob_dist: ",gt_prob_dist.size())
    # print("log_pred_distrib : ",log_pred_prob_distribution.size())
    #Is this masking ?
    # masked_gt_prob_dist = gt_prob_dist[img_mask]
    # masked_log_pred_prob_dist = log_pred_prob_distribution[img_mask]

    # already masked generated ground truth 
    # masking by multiplication 
    masked_gt_prob_dist = gt_prob_dist
    masked_log_pred_prob_dist = log_pred_prob_distribution*img_mask

    # print("masked_log_pred_prob_dist: ",masked_log_pred_prob_dist.size())
    # print("masked_gt_prob_dist: ",masked_gt_prob_dist.size())
    
    loss = 0
    loss = -(masked_log_pred_prob_dist * masked_gt_prob_dist).sum(dim=1, keepdims=True).mean() #check here

    return loss


def model_loss_train(disp_ests, disp_gts, img_masks):
    weights = [1.0, 0.3]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)


def model_loss_test(disp_ests, disp_gts,img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)


#combination of smooth-l1 loss and cross-entropy loss
def model_loss_train_v2(disp_ests, disp_gts, img_masks, max_disp): #level at {1, 1/4}
    weights = [1.0, 0.3]

    pred, pred_4 = disp_ests
    disp_gt, disp_gt_4 = disp_gts
    img_mask, img_mask_4 = img_masks
    
    # print("Disp ests: ",pred_4.size())
    # print("Gt ests: ",disp_gt_4.size())

    loss_l1_1 = weights[0] * F.smooth_l1_loss(pred[img_mask], disp_gt[img_mask], size_average=True) # at size 1
    loss_l1_4 = weights[1] * F.smooth_l1_loss(pred_4[img_mask_4], disp_gt_4[img_mask_4], size_average=True)  # at size 1/4
    
    loss_ce_4 = 0
    # Commenting out -> Gives channel number error
    # Printing out the inputs
    # print("Predicted, GT, img_mask, maxdisp", pred_4.size(),disp_gt_4.size(),img_mask_4.size(),max_disp)
    # loss_ce_4 = weights[1] * ce_based_distribution_loss(pred_4, disp_gt_4, img_mask_4, max_disp//4) #at size 1/4

    loss = loss_l1_1 + loss_l1_4 + loss_ce_4

    return loss


def model_loss_train_v3(disp_ests, disp_gts, img_masks, max_disp): # level at {1, 1/2, 1/4}
    weights = [1.0, 0.3, 1.0]

    pred_up, pred_2, pred_prob_4 = disp_ests
    disp_gt, disp_gt_2, disp_gt_4 = disp_gts
    img_mask, img_mask_2, img_mask_4 = img_masks

    loss1 = weights[0] * F.smooth_l1_loss(pred_up[img_mask], disp_gt[img_mask], size_average=True) # at size 1
    loss2 = weights[1] * F.smooth_l1_loss(pred_2[img_mask_2], disp_gt_2[img_mask_2], size_average=True)  # at size 1/2
    loss3 = weights[2] * ce_based_distribution_loss(pred_prob_4, disp_gt_4, img_mask_4, max_disp/4) #at size 1/4

    loss = loss1 + loss2 + loss3

    return loss
