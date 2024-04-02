import torch
import torch.nn as nn
import torch.nn.functional as F

m = 1
n = 9


# Define a custom convolutional kernel for computing absolute differences
# kernel = torch.tensor([[1, 1, 1, 1, -8, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
kernel = torch.tensor([[1, 1, 1], 
                       [1, -8, 1], 
                       [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Example input tensor (1 batch, 1 channel, 5x5)
input_tensor = torch.tensor([[2, 1, 7, 4, 5],
                             [1, 3, 5, 9, 10],
                             [2, 6, 1, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("Input tensor:")
print(input_tensor.squeeze())

# Instead of getting the summation of all the differences and then absing it
# what we need is to check each differnec if it falls withing threshold
# if so increment P1 counter
# else incrementr P2 counter
# So the end P1 and P2 counter maps are of the HxW size each


unfolded_gtDisp = F.pad(input_tensor,pad=(9//2,9//2,1//2,1//2), mode='constant',value=0) 
print("Padded : ", unfolded_gtDisp)

m1 = m//2
n1 = n//2

unfolded_gtDisp = unfolded_gtDisp.squeeze(0).squeeze(0)
print("Padded : ", n1,m1, unfolded_gtDisp)

for j in range(n1,5+n1):
    for i in range(m1,5+m1):
        p1_count =0
        p2_count =0
        print("value , i , j",unfolded_gtDisp[i,j],i,j)
        # print("r,c variants: ",j,i,j-n1,j+n1,i-m1,i+m1)
        
        for r in range(j-n1,j+n1+1):
            for c in range(i-m1,i+m1+1):
                print("r,c",r,c)
                if(abs(unfolded_gtDisp[c,r]-unfolded_gtDisp[i,j]) < 3):
                    p1_count = p1_count + 1
                else:
                    p2_count = p2_count + 1
                    
        # print("p1 and p2 count at j,i and value: ",p1_count,p2_count,i,j,unfolded_gtDisp[i,j])
# Apply convolution using the custom kernel
# output_tensor = nn.functional.conv2d(input=input_tensor, weight=kernel, padding=(1,1),stride=1)
# output_tensor = nn.functional.conv2d(input=unfolded_gtDisp, weight=kernel, stride=1)

# abs_output_tensor = torch.abs(output_tensor)

# print("\nOutput tensor (differences with neighborhood, before taking absolute value):")
# print(output_tensor.squeeze())
# print("\nOutput tensor (absolute differences with neighborhood):")
# print(abs_output_tensor.squeeze())