
import sys
sys.path.append('../')
sys.path.append('../environment')
#import tensorflow as tf
#from tensorflow.keras import layers
#from env import robot_env
import numpy as np
from pcc_calculation import T1_hole,T2_hole
'''
env = robot_env()
print(env.observation_space.shape[0])
num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states));
inputs = layers.Input(shape=(num_states,))
print(inputs)
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
# Coordinates of the points
points_data = np.array([
    [-0.0371, 0.1386, 0],
    [-0.0049, 0.1386, 0.2566],
    [0.0898, 0.1386, 0.4972],
    [0.2411, 0.1386, 0.7069],
    [0.4396, 0.1386, 0.8727]
])

# Calculate distances between consecutive points
distances = np.linalg.norm(np.diff(points_data, axis=0), axis=1)

# Display the distances
print("Distances between consecutive points:")
for i, distance in enumerate(distances, start=1):
    print(f"l1 {i}-{i + 1}: {distance:.4f}")
'''
   

