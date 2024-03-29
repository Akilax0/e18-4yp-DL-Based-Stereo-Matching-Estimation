import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a custom convolutional kernel for computing absolute differences
kernel = torch.tensor([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Example input tensor (1 batch, 1 channel, 5x5)
input_tensor = torch.tensor([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("Input tensor:")
print(input_tensor.squeeze())

# unfolded_gtDisp = F.pad(input_tensor,pad=(9//2,9//2,1//2,1//2), mode='constant',value=0) 
# print("Padded : ", unfolded_gtDisp)

# Apply convolution using the custom kernel
output_tensor = nn.functional.conv2d(input=input_tensor, weight=kernel, padding=0,stride=1)

abs_output_tensor = torch.abs(output_tensor)

print("\nOutput tensor (differences with neighborhood, before taking absolute value):")
print(output_tensor.squeeze())
print("\nOutput tensor (absolute differences with neighborhood):")
print(abs_output_tensor.squeeze())