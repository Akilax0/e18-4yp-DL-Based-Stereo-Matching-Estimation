import torch.nn as nn
import torch.nn.functional as F
import torch


class MGDLoss(nn.Module):

    """
    PyTorch version of 'Masked Generative Distillation'

    Args:
        student_channels(int): Number of channels in the student's feature map
        teacher_channels(int): Number of channels in the teacher's feature map
        name (str): the loss name of the layer 
        alpha_mgd (float, optional): Weight of dis loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Default to 0.5

    Edit: 
        Here super and channel alignment removed
        what does the @DISTLL_LOSSES.register__module() do?

    ref: Z. Yang, Z. Li, M. Shao, D. Shi, Z. Yuan, and C. Yuan,
    “Masked Generative Distillation,” arXiv.org, May 03, 2022
    
    """
    
    def __init__(self,
                student_channels,
                teacher_channels,
                name,
                lambda_mgd=0.15,
                ):
        

        self.lambda_mgd = lambda_mgd
        
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None


        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)
        )


    def forward(self,
               preds_S,
               preds_T):
    
        """
        Forward function.

        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        
        assert preds_S.shape[-2:] == preds_T.shape[-2:]


        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        # check the usage here
        loss_mse = nn.MSELoss(reduction='sum')
        
        N, C, H, W = preds_T.shape

        print(preds_S.device)
        device = preds_S.device
        mat = torch.rand((N,C,1,1)).to(device)

        # mask generation
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        # mask aligned student 
        masked_feat = torch.mul(preds_S, mat)
        
        # Genearate feature from student to be compared with teacher
        new_feat = self.generation(masked_feat)

        # calculate distilation loss
        # check the implementation here for distillation loss
        # dis_loss = loss_mse(new_feat, preds_T)/N
        dis_loss = F.mse_loss(new_feat,preds_T)

        return dis_loss

