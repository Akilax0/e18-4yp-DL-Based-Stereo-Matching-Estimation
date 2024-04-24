import torch.nn.functional as F
import torch


def dec(student,mask):
    '''
    adaptive decoder module. 

    Includes the following, 
    - spatial alignment module
    - stacked transformer decoder layers
    - spatial recovery module
    
    mask - mask token
    

    The input student is already aligned and masked. 
    Then goes to SAM

    SAM :

    inputs = (Hi,Wi,CT)

    checks if greater than H/32 and W/32 
    if so conv with stride p else stride 1/p
    where(p = Hi/(H/32))
    upsample using nearest neighbour
    
    outputs = (H/32, W/32 , CT)
    


    decoder :
    # fill this out

    positional embedding

    Normalization layer to divide top 
    q
    k
    v
    MHSA

    add before

    Norm 
    FFN
    
    add before


    SRM : 

    input (H/32, W/32,CT)

    linear opeation output (H/32,W/32,p^2 x CT)

    reshape if needed to match teacher size

    outputs = (Hi,Wi,CiT) 

    MSE loss to calculate the error with teacher. 



    '''

    out = 0
    
    # mask student with mask token

    
    return out


def masked(student):

    '''
    
    Random patchwise masker
    
    '''

    stu_mask = 0

    return stu_mask


