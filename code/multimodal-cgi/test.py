import torch
import torch.nn as nn
import torch.nn.functional as F

#transfer gt disparity map to probability distribution map
m = 1
n = 9 # m x n neighborhood to determine whether the centering pixel is edge or not
th1 = 1 # threshold to determine whether it is edge or not
th2 = 1 # threshold to determine p1-p2 clustering
alpha = 0.8
    


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
print(input_tensor)

# Instead of getting the summation of all the differences and then absing it
# what we need is to check each differnec if it falls withing threshold
# if so increment P1 counter
# else incrementr P2 counter
# So the end P1 and P2 counter maps are of the HxW size each


# Define the neighborhood size
neighborhood_size = (1, 9)  # Assuming a 1x9 neighborhood

# Pad the tensor to ensure every pixel becomes the central pixel of a neighborhood
padding = (neighborhood_size[0] // 2, neighborhood_size[1] // 2)
padded_tensor = F.pad(input_tensor, (padding[1], padding[1], padding[0], padding[0]), mode='constant', value=0)

print("Padded tensorr: ",padded_tensor )

padded_tensor = padded_tensor.squeeze(0).squeeze(0)
print("padded tensor squeezed: ",padded_tensor.size())

# Unfold the padded tensor to extract neighborhoods
unfolded_tensor = padded_tensor.unfold(0, neighborhood_size[0], 1).unfold(1,9,1)
print("unfolded tensor: " , unfolded_tensor.size())


# Extract central pixels from each neighborhood
central_pixels = unfolded_tensor[:, :, 0, neighborhood_size[1] // 2].unsqueeze(-1).unsqueeze(-1)
print("Central pixels size: ", central_pixels.size())

# Calculate differences between central pixels and the rest of the neighborhood elements
differences = torch.abs(unfolded_tensor - central_pixels)
print("Differences size: ",differences.size())

print("Differences: ",differences.size())

thresh = 3 

p1_p2_cluster = torch.where(differences< thresh,1, 0 )
print("p1_cluster :",p1_p2_cluster.size() )

p1_sum = torch.sum(p1_p2_cluster, dim=3).squeeze(-1)
print("p1_sum: ",p1_sum.size())

p1_points = torch.zeros_like(input_tensor)
p1_points = p1_sum.unsqueeze(0).unsqueeze(0) 
print("p1 cluster: ",p1_points)

p2_points = m*n - p1_points

print("p2_points :" , p2_points)


# adding small value to p2 to remove divide by 0 error
p2_points = p2_points + 0.0001
print("p2_count",p2_points.eq(0).any())

# # p2_ount contains 0 resulting in NaNs 
# # what can we do here?
# mu2 = torch.sum(p1_p2_cluster * neighborhood, dim=1, keepdim=True) / p2_count # pay attention divided by 0

print("p1_p2_cluster size :",p1_p2_cluster.size())    
print("unfolded tensor size :",unfolded_tensor.size())    
print("p2_points size",p2_points.size())
print("multiplication ", (p1_p2_cluster*unfolded_tensor).size())


# Need to check calculating mu2  
# mu2 = torch.sum(p1_p2_cluster * unfolded_tensor, dim=5, keepdim=True).squeeze(-1).squeeze(-1)  / p2_points
mu2 = torch.sum(p1_p2_cluster * unfolded_tensor, dim=5, keepdim=True)  / p2_points

print("mu2 size: ",mu2.size())

# As in paper
w = alpha + (p1_points - 1) * (1-alpha) * (m*n-1)
print("w size : ",w.size())

# print("p1_count:" ,torch.sum(p1_cluster))

# Print the results
# for i in range(differences.size(0)):
#     for j in range(differences.size(1)):
#         central_pixel = central_pixels[i, j].item()
#         neighborhood_differences = differences[i, j]
#         # print("neighbourhood: ",unfolded_tensor[:,:,0,neighborhood_size[1]//2])
#         print(f"Neighborhood at position ({i}, {j}) - Central pixel: {central_pixel:.2f}")
#         print("Differences:")
#         print(neighborhood_differences)



# unfolded_gtDisp = F.pad(input_tensor,pad=(9//2,9//2,1//2,1//2), mode='constant',value=0) 
# print("Padded : ", unfolded_gtDisp)

# m1 = m//2
# n1 = n//2

# unfolded_gtDisp = unfolded_gtDisp.squeeze(0).squeeze(0)
# print("Padded : ", n1,m1, unfolded_gtDisp)

# for j in range(n1,5+n1):
#     for i in range(m1,5+m1):
#         p1_count =0
#         p2_count =0
#         print("value , i , j",unfolded_gtDisp[i,j],i,j)
#         print("r,c variants: ",j,i,j-n1,j+n1,i-m1,i+m1)
        
#         for r in range(j-n1,j+n1+1):
#             for c in range(i-m1,i+m1+1):
#                 # print("r,c",r,c)
#                 if(abs(unfolded_gtDisp[c,r]-unfolded_gtDisp[i,j]) < 3):
#                     p1_count = p1_count + 1
#                 else:
#                     p2_count = p2_count + 1
                    
        # print("p1 and p2 count at j,i and value: ",p1_count,p2_count,i,j,unfolded_gtDisp[i,j])
        


# Apply convolution using the custom kernel
# output_tensor = nn.functional.conv2d(input=input_tensor, weight=kernel, padding=(1,1),stride=1)
# output_tensor = nn.functional.conv2d(input=unfolded_gtDisp, weight=kernel, stride=1)

# abs_output_tensor = torch.abs(output_tensor)

# print("\nOutput tensor (differences with neighborhood, before taking absolute value):")
# print(output_tensor.squeeze())
# print("\nOutput tensor (absolute differences with neighborhood):")
# print(abs_output_tensor.squeeze())