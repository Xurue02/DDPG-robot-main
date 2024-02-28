
# from configuration space (k, length) to task space (x,y)

# % import necessary libraries
import sys # to include the path of the package
sys.path.append('../')
#from continuum_robot.utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
from kinematics.pcc_forward import trans_matrix,multiple_trans_matrix,two_section_robot,arc1_point,arc2_point,visual,cable_len
#from datetime import datetime

#print("hello")
## Enter two k values within the range
k1=1;
k2=1;
phi1=0
phi2=0
s1_hole = np.radians([105,225,345])
s2_hole = np.radians([75,195,315])
d = 35.285/246
l = 1 # meter, same length for every segment
if k1 > (5*math.pi)/3 or k1 < -(5*math.pi)/3:
    print("Please enter the First Curvature values between -5.235 and 5.235")
    k1 = 0;

elif k2 > (5*math.pi)/3 or k2 < -(5*math.pi)/3:
    print("Please enter the Second Curvature values between -5.235 and 5.235")
    k2 = 0;
else:
    print("Curvature Values for Each Segment are Appropriate")


'''
l = 0.246 # meter, same length for every segment
## Randomly choose two k values within the range
##k_range = (-(5 * math.pi) / 3, (5 * math.pi) / 3)
##k_range = (-16,16)
##k1, k2 = np.random.uniform(*k_range, size=2)
# Constraint for the curvature
##k1 = max(-(5 * math.pi) / 3, min(k1, (5 * math.pi) / 3))
##k2 = max(-(5 * math.pi) / 3, min(k2, (5 * math.pi) / 3))

s1_hole = np.radians([105,225,345])
s2_hole = np.radians([75,195,315])
d = 0.035286 #35.286m from hole to the center of the robot

k1 = np.random.uniform(low=-10, high=16)
k2 = np.random.uniform(low=-10, high=16)
print(f"Randomly chosen k1: {k1}")
print(f"Randomly chosen k2: {k2}")

## Randomly choose two phi values within the range
## Set the seed for reproducibility
np.random.seed(42)
# Generate two random values for phi in radians within the range from -180 to 180 degrees
phi1 = np.radians(np.random.uniform(low=-180, high=180))
phi2 = np.radians(np.random.uniform(low=-180, high=180))

print(f"Randomly selected phi1: {phi1:.4f} radians")
print(f"Randomly selected phi2: {phi2:.4f} radians")

'''

# segment 1
T1 = trans_matrix(k1,l,phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F'); #reshape to 4*4 matrix

# turn central point to three holes
T1_hole = arc1_point(T1,s1_hole,d) #15 arrays, each of(hole1, hole2,hole3,1)
#T1_1,T1_2,T1_3 = T1_hole[-1][:3] #Section1 tip holes
#print('T1 transmatrix\n',T1);
#print('T1 holes\n',T1_hole);
#print('T1 tipss\n',T1_1,T1_2,T1_3);
#print(T1[:, 12])
#print('num within section1',len(T1))
#print('T1_tip\n',T1_tip);
#print(T1[0,12],T1[0,13],T1[1,0],T1[1,12],T1[1,13])


# segment 2
T2_cc = trans_matrix(k2,l,phi2);#get reshaped transformation matrix of the section 2 
T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
T2_tip = np.reshape(T2[len(T2)-1,:],(4,4),order='F');# reshape to 4*4 matrix

# turn central point to three holes
T2_hole = arc2_point(T2_cc,T2,s2_hole,d)  #30 arrays, each of(hole4, hole5,hole6,1)
#T2_1,T2_2,T2_3 = T2_hole[-1][:3]
#print('T2 holes\n',T2_hole);
#print('T1 tipss\n',T2_1,T2_2,T2_3);
#print('T2 transmatrix\n',T2);
#print('tip of the robot\n',T2_tip);


l6_len = cable_len(T1_hole,T2_hole)
print("cable length of first segment are",l6_len) # ([2.02887214168416], [2.154659209378555], [1.6852154818079672])
robot_3d = visual (T1,T1_hole,T2,T2_hole)
'''
# Plotting First Section
ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=2, marker='o')

# Reshape and Plot Holes for First Section
for i in range(3):
    holes_reshaped = np.array(T1_hole).reshape(5, 3, 4)
    ax.plot(holes_reshaped[:, i, 0], holes_reshaped[:, i, 1], holes_reshaped[:, i, 2], color='blue', linewidth=3, marker='o')

# Plotting Second Section
ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=2, marker='o')

# Reshape and Plot Holes for Second Section
for i in range(3):
    holes_reshaped = np.array(T2_hole).reshape(5, 3, 4)
    ax.plot(holes_reshaped[:, i, 0], holes_reshaped[:, i, 1], holes_reshaped[:, i, 2], color='red', linewidth=3, marker='o')


# Plot points for T1
ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=2, marker='o')
T1_holes_reshaped = np.array(T1_hole).reshape(5, 3, 4)
print(T1_holes_reshaped)
l1_x = T1_holes_reshaped[:, 0, 0].flatten()
l1_y = T1_holes_reshaped[:, 0, 1].flatten()
l1_z = T1_holes_reshaped[:, 0, 2].flatten()

l2_x = T1_holes_reshaped[:, 1, 0].flatten()
l2_y = T1_holes_reshaped[:, 1, 1].flatten()
l2_z = T1_holes_reshaped[:, 1, 2].flatten()

l3_x = T1_holes_reshaped[:, 2, 0].flatten()
l3_y = T1_holes_reshaped[:, 2, 1].flatten()
l3_z = T1_holes_reshaped[:, 2, 2].flatten()

# Plotting
ax.plot(l1_x, l1_y, l1_z ,color='blue', linewidth=3, marker='o')
ax.plot(l2_x, l2_y, l2_z ,color='blue', linewidth=3, marker='o')
ax.plot(l3_x, l3_y, l3_z ,color='blue', linewidth=3, marker='o')

ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=2, marker='o')
T2_holes_reshaped = np.array(T2_hole).reshape(5, 3, 4)
print(T1_holes_reshaped)
l4_x = T2_holes_reshaped[:, 0, 0].flatten()
l4_y = T2_holes_reshaped[:, 0, 1].flatten()
l4_z = T2_holes_reshaped[:, 0, 2].flatten()

l5_x = T2_holes_reshaped[:, 1, 0].flatten()
l5_y = T2_holes_reshaped[:, 1, 1].flatten()
l5_z = T2_holes_reshaped[:, 1, 2].flatten()

l6_x = T2_holes_reshaped[:, 2, 0].flatten()
l6_y = T2_holes_reshaped[:, 2, 1].flatten()
l6_z = T2_holes_reshaped[:, 2, 2].flatten()

# Plotting
ax.plot(l4_x, l4_y, l4_z ,color='red', linewidth=3, marker='o')
ax.plot(l5_x, l5_y, l5_z ,color='red', linewidth=3, marker='o')
ax.plot(l6_x, l6_y, l6_z ,color='red', linewidth=3, marker='o')



l=[1,1];

#Tip_of_Rob = two_section_robot(k1,k2,l,phi1,phi2)
#print('tip of robo should be same as tip of the robot as above\n',Tip_of_Rob)
#x,y,z = np.array([Tip_of_Rob[0,3],Tip_of_Rob[1,3],Tip_of_Rob[2,3]])
#print('x,y,z are\n',x,y,z)



# Plot the 3D diagram python pcc_calculation.py
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot points for T1
ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=3, marker='o')
# Plot points for T2
ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=3, marker='o')

#end-effector of the robot
ee = len(T2) - 1 
ax.text(T2[ee, 12], T2[ee, 13], T2[ee, 14], f'({T2[ee, 12]:.2f}, {T2[ee, 13]:.2f}, {T2[ee, 14]:.2f})', fontsize=8)

# add k and phi values on diagram
ax.text(T1[-1, 12], T1[-1, 13], T1[-1, 14], f'k1={k1:.2f},\n phi1={phi1:.4f}', fontsize=8, ha='right', va='bottom')
ax.text(T2[-1, 12], T2[-1, 13], T2[-1, 14], f'k2={k2:.2f},\n phi2={phi2:.4f}', fontsize=8, ha='right', va='bottom')


# Set labels and title
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3D Plot of Continuum Robot Forward Kinematics")
ax.legend(loc="best")
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.savefig('../figures/3d_robot/tip.png')
plt.show()
'''
# %%
