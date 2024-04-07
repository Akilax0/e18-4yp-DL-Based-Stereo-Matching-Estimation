import torch
import torch.nn as nn
import torch.nn.functional as F

m = 1
n = 9

# Define a custom convolutional kernel for computing absolute differences
# kernel = torch.tensor([[1, 1, 1, 1, -8, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# kernel = torch.tensor([[1, 1, 1], 
#                        [1, -8, 1], 
#                        [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# # Example input tensor (1 batch, 1 channel, 5x5)
# input_tensor = torch.tensor([[2, 1, 7, 4, 5],
#                              [1, 3, 5, 9, 10],
#                              [2, 6, 1, 14, 15],
#                              [16, 17, 18, 19, 20],
#                              [21, 22, 23, 24, 25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


kernel = torch.randn((10,1,1,9))
input_tensor = torch.randn((10,1,64,128))

print("kernel :",kernel.size())
print("Input tensor:",input_tensor.size())
# print(input_tensor)

mean_gt = F.conv2d(input_tensor, kernel, padding=(m // 2, n // 2))
# mean_gt = F.conv2d(input_tensor, kernel)
print("mean_gt",mean_gt.size())
# print(mean_gt)

# mean_gt /= m * n