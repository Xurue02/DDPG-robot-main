import sys # to include the path of the package
sys.path.append('../') # the kinematics functions are here 

import gym                      # openai gym library
import numpy as np              # numpy for matrix operations
import math                     # math for basic calculations
from gym import spaces          # "spaces" for the observation and action space
import matplotlib.pyplot as plt # quick "plot" library
from matplotlib.animation import FuncAnimation # make animation
from kinematics.pcc_forward import trans_matrix,multiple_trans_matrix,two_section_robot,arc1_point,arc2_point,visual,cable_len,get_points
from visualspaces import visualspaces

class robot_env(gym.Env):
    def __init__(self):

        self.l1 = 0.24600;               # first segment of the robot in meters
        self.l2 = 0.24600;               # second segment of the robot in meters
        self.l_l = self.l1+self.l2
        self.stop = 0               # variable to make robot not move after exeeding max, min general k value
        self.l = [self.l1, self.l2]           # stores the length of each segment of the robot
        self.s1_hole = np.radians([105,225,345])
        self.s2_hole = np.radians([75,195,315])
        self.d = 0.035286            # distance in meter from the hole to the center of backbone 
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_k = [0,0]        # initializes the start curvatures for the two segments
        self.start_phi = [0,0]      # initializes the start phi for the two segments
        self.time = 0               # to count the time of the simulation
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.position_dic = {'Section1': {'x':[],'y':[],'z':[]}, 'Section2': {'x':[],'y':[],'z':[]}}

        self.cable_length_min = 0.3  # Adjust based on your robot's constraints
        self.cable_length_max = 1.0  # Adjust based on your robot's constraints
        self.cable_length_change_max = 0.075  # Maximum change in cable lengths for each action.0.05
        #self.cab_lens = list(self.cab_lens)
        # Initialize cable lengths
        #self.cable_lengths = np.array([0.492, 0.492, 0.492, 0.492, 0.492, 0.492])

        # Define observation and action spaces
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32)
        low = np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.cable_length_change_max, high=self.cable_length_change_max, shape=(6,), dtype=np.float32)
        self.observation_space = visualspaces()
 
    def reset(self): 

        # Random state of the robot 
        ## Randomly choose two phi values within the range
        ## Set the seed for reproducibility
        np.random.seed(42)
        # Generate two random values for phi in radians within the range from -180 to 180 degrees
        '''
        self.phi1 = np.radians(np.random.uniform(low=-180, high=180))
        self.phi2 = np.radians(np.random.uniform(low=-180, high=180))

        # Random curvatures 
        self.k1 = np.random.uniform(low=-0, high=1.8)
        self.k2 = np.random.uniform(low=0, high=1.8)
        #self.cable_lengths = np.array([0.492, 0.492, 0.492, 0.492, 0.492, 0.492]) 
      

        # Random target point
        self.target_phi1 = np.radians(np.random.uniform(low=-180, high=180))
        self.target_phi2 = np.radians(np.random.uniform(low=-180, high=180))
        # (Random curvatures are given so that forward kinematics equation will generate random target position)
        self.target_k1 = np.random.uniform(low=0, high=1.8) # 6.2 # np.random.uniform(low=-4, high=16)
        self.target_k2 = np.random.uniform(low=0, high=1.8) # 6.2 # np.random.uniform(low=-4, high=16)

        '''
        #manually set
        self.phi1 = 0
        self.phi2 = 0
        self.k1 = 0.001
        self.k2 = 0.001

        self.target_phi1 = np.radians(0)
        self.target_phi2 = np.radians(90)
        self.target_k1 = -1.5 
        self.target_k2 = 1.5
        ## 0.24254401377535303, 0.23670913754459977, 0.2584851920936469, 0.5309091514774634, 0.5627915081140512, 0.5760698989260247)
        

        # pcc calculation Initail point
        Tip_of_Rob = two_section_robot(self.k1,self.k2,self.l,self.phi1,self.phi2) 
        x,y,z = np.array([Tip_of_Rob[0,3],Tip_of_Rob[1,3],Tip_of_Rob[2,3]]) # Extract the x,y and z coordinates of the tip

        # pcc calculation Target point
        Tip_target = two_section_robot(self.target_k1,self.target_k2,self.l,self.target_phi1,self.target_phi2) # Generate the target point for the robot
        goal_x,goal_y,goal_z = np.array([Tip_target[0,3],Tip_target[1,3],Tip_target[2,3]]) # Extract the x and y coordinates of the target
        self.state = x,y,z,goal_x,goal_y,goal_z # Update the state of the robot

        self.goal_x,self.goal_y,self.goal_z = goal_x,goal_y,goal_z
        #print(goal_x,goal_y,goal_z)


        self.last_u = None
        return self._get_obs()
    
    def _get_obs(self):
        x,y,z,goal_x,goal_y,goal_z = self.state
        return np.array([x,y,z,goal_x,goal_y,goal_z],dtype=np.float32)
    
    def cab_len(self):

        T1 = trans_matrix(self.k1,self.l[0],self.phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F');  
        T1_hole = arc1_point(T1,self.s1_hole,self.d) #15 arrays, each of(hole1, hole2,hole3,1)
        
        T2_cc = trans_matrix(self.k2,self.l[1],self.phi2);#get reshaped transformation matrix of the section 2 
        T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
        T2_hole = arc2_point(T2_cc,T2,self.s2_hole,self.d)  #30 arrays, each of(hole4, hole5,hole6,1)
        l6_len = cable_len(T1_hole,T2_hole)
        #self.cab_lens_1, self.cab_lens_2, self.cab_lens_3, self.cab_lens_4, self.cab_lens_5, self.cab_lens_6 = l6_len[:6]
        self.cab_lens = l6_len[:6]
        self.cab_lens = list(self.cab_lens)

        # target len
        target_T1 = trans_matrix(self.target_k1,self.l[0],self.target_phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        target_T1_tip = np.reshape(target_T1[len(target_T1)-1,:],(4,4),order='F');  
        target_T1_hole = arc1_point(target_T1,self.s1_hole,self.d) #15 arrays, each of(hole1, hole2,hole3,1)
        
        target_T2_cc = trans_matrix(self.target_k2,self.l[1],self.target_phi2);#get reshaped transformation matrix of the section 2 
        target_T2 = multiple_trans_matrix(target_T2_cc,target_T1_tip); # multiply T1 and T2 to get the robot transformation matrix
        target_T2_hole = arc2_point(target_T2_cc,target_T2,self.s2_hole,self.d)  #30 arrays, each of(hole4, hole5,hole6,1)
        target_l6_len = cable_len(target_T1_hole,target_T2_hole)

        self.target_cab_lens = target_l6_len[:6]
        

        return l6_len,target_l6_len
    
    
    
    def step(self, action):
        # 1. Calculate new position based on action
        # 2. Calculate current distance from goal
        # 3. Compute reward based on distance change
        # x,y,z,goal_x,goal_y,goal_z = self.state
        self.past_distance = 0.0
        action = np.clip(action, -self.cable_length_change_max, self.cable_length_change_max) # Clip the input to the range of the -0.075,0.075
        #for i in range(0,6):
        #    while action[i]> 0.1 or u[i] < -0.1:
        #        action[i]=u[i]/10

        '''
        for i in range(0,5):#0 1 2 3 4 5 
            self.cab_lens[i] = self.cab_lens[i]+action[i]
            
            #if i in range(0,4):
        self.cab_lens[0] = np.clip(self.cab_lens[0], self.l1-self.cable_length_change_max, self.l1+self.cable_length_change_max)
        self.cab_lens[1] = np.clip(self.cab_lens[1], self.l1-self.cable_length_change_max, self.l1+self.cable_length_change_max)
        self.cab_lens[2] = np.clip(self.cab_lens[2], self.l1-self.cable_length_change_max, self.l1+self.cable_length_change_max)
        self.cab_lens[3] = np.clip(self.cab_lens[3], self.l_l-self.cable_length_change_max, self.l_l+self.cable_length_change_max)
        self.cab_lens[4] = np.clip(self.cab_lens[4], self.l_l-self.cable_length_change_max, self.l_l+self.cable_length_change_max)
        self.cab_lens[5] = np.clip(self.cab_lens[5], self.l_l-self.cable_length_change_max, self.l_l+self.cable_length_change_max)
        '''
        for i in range(6):
            if i < 3:
                clip_range = (self.l1 - self.cable_length_change_max, self.l1 + self.cable_length_change_max)
            else:
                clip_range = (self.l_l - self.cable_length_change_max, self.l_l + self.cable_length_change_max)
            self.cab_lens[i] = np.clip(self.cab_lens[i] + action[i], *clip_range)

        #print('######',action)    
        new_x,new_y, new_z = get_points(self.cab_lens)
        
        # Calculate current distance from goal
        self.current_distance = math.sqrt((self.goal_x - new_x) ** 2 +
                                     (self.goal_y - new_y) ** 2 +
                                     (self.goal_z - new_z) ** 2)
        
        cable_change_penalty = np.sum(np.abs(action))
        #self.current_distance = math.sqrt(((self.goal_x-new_x)**2)+((self.goal_y-new_y)**2)+((self.goal_z-new_z)**2)) # Calculate the error squared
        #reward = -1 * self.current_distance 
        
        reward = -1 * self.current_distance - 0.1 * cable_change_penalty
        
        if reward <= 0.01:
        #if reward <= -0.01:
            done = True
        else:
            done = False
            #reward = -10
        '''
        if self.current_distance <= 0.01:
            # Goal reached
            reward = -1 * self.current_distance
            done = True
        else:
            # Compute reward based on distance change
            #distance_rate = self.past_distance - self.current_distance
            #reward = -10 * distance_rate
            reward = -1 * self.current_distance - 0.1 * cable_change_penalty
            done = False
        '''
        # Update state variables
        self.state = (new_x, new_y, new_z,self.goal_x, self.goal_y, self.goal_z)
        self.past_distance = self.current_distance
        
        return self.state, reward, done, {}