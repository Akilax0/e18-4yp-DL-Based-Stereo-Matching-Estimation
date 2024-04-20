import torch
import torch.nn as nn
import torch.nn.functional as F

max_disp = 20
prob = torch.rand((1,max_disp,64,128))
# prob = torch.abs(prob)

h1 = 3
w1 = 2

b,d,h,w = prob.size()

max_probs , max_indices = torch.max(prob, dim=1)
left_bound = torch.zeros_like(max_indices)
right_bound = torch.zeros_like(max_indices)

# print("All dimensions prob, max_probs, max_indices, left_bound, right_bound : ",prob.size(), max_probs.size(),max_indices.size(), left_bound.size(),right_bound.size())
# print("Max Indices: ",max_indices)

batch_prob = prob[:,:,:,:]

# print("prob: ",batch_prob)

# Caluclating right bounds
# print("current channel: ", batch_prob[:,:-1,:,:])
# print("next channel: ", batch_prob[:,1:,:,:])
# print("max indices: ",max_indices)

right_bounds = torch.ones_like(prob)
right_bounds[:,:-1,:,:] = batch_prob[:,:-1,:,:]<batch_prob[:,1:,:,:]
# print("Right Bounds: ",right_bounds)

# Find the positions of FAlSE 
true_locations = torch.nonzero(right_bounds == True)
true_locations_r = torch.flip(true_locations,dims=[0])

# Setting right bound to max disp 
right_bound = right_bound + (max_disp-1)

t0 = true_locations_r[:,0]
t1 = true_locations_r[:,1]
t2 = true_locations_r[:,2]
t3 = true_locations_r[:,3]

update_values = right_bound[t0,t2,t3]

# print("updated_values: ",update_values,t0,t1,t2,t3)

differences = t1-max_indices[t0,t2,t3]
mask = differences >= 0
update_values = update_values[mask]
t0 = t0[mask]
t1 = t1[mask]
t2 = t2[mask]
t3 = t3[mask]

# print("updated_values: ",update_values,t0,t1,t2,t3)

# Need to figure out a way to do this in one go 
# Rather than repeatedly updating need to select the min so far
# update_values = torch.minimum(t1,update_values) * ((t1 - max_indices[t0,t2,t3])>=0) 
# Adding and removing one to differnetiate between mask and actual 0
update_values = torch.minimum(t1,update_values) #+ 1) * ((t1-max_indices[t0,t2,t3]>=0))
# print("updated value: ",update_values)

update_values = torch.where(update_values > 0, update_values, right_bound[t0, t2, t3])
# print("updated value: ",update_values)

right_bound[t0,t2,t3] = update_values


# Calculating Left bounds
left_bounds = torch.ones_like(prob)
left_bounds[:,1:,:,:] = batch_prob[:,1:,:,:]<batch_prob[:,:-1,:,:]
# print("Left Bounds: ",left_bounds)

# Find the positions of FAlSE 
true_locations_l = torch.nonzero(left_bounds == True)
# print("True Locations (Right): ",len(true_locations))

# # print("Left bound: ",left_bound)
t0 = true_locations_l[:,0]
t1 = true_locations_l[:,1]
t2 = true_locations_l[:,2]
t3 = true_locations_l[:,3]

update_values = left_bound[t0,t2,t3]
# print("updated_values: ",update_values,t0,t1,t2,t3)

differences = max_indices[t0,t2,t3] - t1
mask = differences >= 0
update_values = update_values[mask]
t0 = t0[mask]
t1 = t1[mask]
t2 = t2[mask]
t3 = t3[mask]
# print("updated_values: ",update_values,t0,t1,t2,t3)

update_values = torch.maximum(t1,update_values)
#* (( max_indices[t0,t2,t3] - t1)>=0) 
update_values = torch.where(update_values > 0, update_values, left_bound[t0, t2, t3])

left_bound[t0,t2,t3] = update_values

print("==================right======================")
for i in range(len(true_locations_r)):
    if true_locations_r[i][2]==h1 and true_locations_r[i][3]==w1:
        print("true location index , location : ",i,true_locations_r[i])
print("==================left======================")
for i in range(len(true_locations_l)):
    if true_locations_l[i][2]==h1 and true_locations_l[i][3]==w1:
        print("true location index , location : ",i,true_locations_l[i])
print("prob: ",prob[0,:,h1,w1])
print("right bounds: ",right_bounds[0,:,h1,w1])
print("left bounds: ",left_bounds[0,:,h1,w1])
print("right bound: ",right_bound[0,h1,w1])
print("left bound: ",left_bound[0,h1,w1])
print("max indices : ",max_indices[0,h1,w1])
print("max probs: ",max_probs[0,h1,w1])

# print("True Locations (Right): ",true_locations)
# =======================================================================================

# # prob [20,48,64,128]
# # max_prob [20,64,128]
# # max_indices [20,64,128]
# # left_bound [20,64,128]
# # right_bound [20,64,128]
# for c in range(prob.size(0)):
#     for i in range(prob.size(2)):
#         for j in range(prob.size(3)):

#             # ma_d = max_probs[c,i,j]
#             ma_x = max_indices[c,i,j]
#             right = ma_x
#             print("max index: ",ma_x)

#             for d in range(ma_x+1,prob.size(1)):
#                 if(prob[c,d,i,j]<=prob[c,d-1,i,j]):
#                     right = d
#                 else:
#                     break;

#             right_bound[:,i,j] = right

            # left = ma_x

            # for d in range(ma_x-1,0,-1):
            #     if(prob[:,d,i,j]<=prob[:,d+1,i,j]):
            #         left = d
            #     else:
            #         break;

            # left_bound[:,i,j] = left
            
            # print("left_bound, right_bound, right, left: ",left_bound.size(), right_bound.size(), left.size(), right.size())
            

# ======================================================================================================
