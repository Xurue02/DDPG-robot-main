
import sys
sys.path.append('../')
sys.path.append('../environment')
import tensorflow as tf
from tensorflow.keras import layers
from env import robot_env
import numpy as np
#from pcc_calculation import T1_hole,T2_hole

env = robot_env()

num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states));
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions));
print('actionspace',env.action_space)

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound));
print("Min Value of Action ->  {}".format(lower_bound));





'''
def cable_len(T1_hole,T2_hole):
    l1_len, l2_len, l3_len, l4_len, l5_len, l6_len = 0, 0, 0, 0, 0, 0
    T1_reshaped = np.array(T1_hole).reshape(5, 3, 4)
    for i in range(4):
        l1_len += np.linalg.norm(T1_reshaped[i+1, 0, :3] - T1_reshaped[i, 0, :3])
        l2_len += np.linalg.norm(T1_reshaped[i+1, 1, :3] - T1_reshaped[i, 1, :3])
        l3_len += np.linalg.norm(T1_reshaped[i+1, 2, :3] - T1_reshaped[i, 2, :3])

    T2_reshaped = np.array(T2_hole).reshape(10, 3, 4)
    #T2_reshaped = np.delete(T2_reshaped, 4, axis=0)
    for i in range(9):        
        l4_len += np.linalg.norm(T2_reshaped[i+1, 0, :3] - T2_reshaped[i, 0, :3])
        l5_len += np.linalg.norm(T2_reshaped[i+1, 1, :3] - T2_reshaped[i, 1, :3])
        l6_len += np.linalg.norm(T2_reshaped[i+1, 2, :3] - T2_reshaped[i, 2, :3])
    
    return l1_len,l2_len,l3_len, l4_len, l5_len, l6_len

l6_len = cable_len(T1_hole,T2_hole)
print("Total distance:", l6_len)

    
    #print('T1_reshaped',T1_reshaped)
    #print('T2_reshaped',T2_reshaped)

'''
   

