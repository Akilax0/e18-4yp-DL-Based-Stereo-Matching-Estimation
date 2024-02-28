import torch.nn as nn
import torch.nn.functional as F
import torch


def align(student,student_channels,teacher_channels):
    # Given two tensors of different channel numbers 
    # align the two with a 1x1 kernel
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = student.device
    # print("device: ",device)

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

def reducer(student,student_channels):
    # Given two tensors of different channel numbers 
    # align the two with a 1x1 kernel
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = student.device
    # print("device: ",device)

    # print("student and teacher channels",student_channels, teacher_channels)
    # print("Types: ",student.type())
    if student_channels!= student.size()[1] and len(student.size())==4:
        m = nn.Conv2d(student.size()[1], student_channels, kernel_size=1, stride=1, padding=0).to(device)
    elif student_channels!= student.size()[1] and len(student.size())==5:
        m = nn.Conv3d(student.size()[1], student_channels, kernel_size=(1,1,1), stride=1, padding=0).to(device)
    else:
        m = None
        return student

    return m(student)

def get_dis_loss(preds_S, preds_T,student_channels, teacher_channels, lambda_mgd=0.15):


    N, C, H, W = preds_T.shape

    device = preds_S.device
    
    # print("device: " ,device)

    generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)).to(device)


    mat = torch.rand((N,C,1,1)).to(device) 
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
    # print("dis_loss : " ,dis_loss.size())

    return dis_loss,new_feat

def get_dis_loss_3D(preds_S, preds_T,student_channels, teacher_channels, lambda_mgd=0.15):


    N, C, D, H, W = preds_T.shape

    device = preds_S.device
    
    # print("device: " ,device)

    generation = nn.Sequential(
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv3d(teacher_channels, teacher_channels, kernel_size=3, padding=1)).to(device)


    mat = torch.rand((N,C,1,1,1)).to(device) 
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

    return dis_loss,new_feat