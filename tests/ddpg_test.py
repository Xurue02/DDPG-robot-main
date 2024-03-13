# %%
import sys
sys.path.append('../')
sys.path.append('../environment')
sys.path.append('../tensorflow')
sys.path.append('../tests')

import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
from env2 import robot_env
from ddpg import OUActionNoise,policy
plt.style.use('../continuum_robot/plot.mplstyle')
from plotsfunc import *
from matplotlib import animation
# %matplotlib notebook
from IPython import display

storage = {store_name: {} for store_name in ['error', 'pos', 'k','reward']}
storage['error']['error_store'] = []
storage['error']['x'] = []
storage['error']['y'] = []
storage['error']['z'] = []

storage['pos']['x'] = []
storage['pos']['y'] = []
storage['pos']['z'] = []

#storage['k']['k1'] = []
#storage['k']['k2'] = []

storage['cable length']['l1'] = []
storage['cable length']['l2'] = []
storage['cable length']['l3'] = []
storage['cable length']['l4'] = []
storage['cable length']['l5'] = []
storage['cable length']['l6'] = []



storage['reward']['value'] = []
storage['reward']['effectiveness'] = []

episode_number = 5
for _ in range(episode_number):
    print("hellooooo")
    env2 = robot_env() # initialize environment

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))

    state = env2.reset() # generate random starting point for the robot and random target point.
    env2.time = 0
    #env2.start_k = [env2.k1, env2.k2] # save starting ks
    initial_state = state[0:3] # state = x,y,z,goal_x,goal_y,goal_z

    env2.render_init() # uncomment for animation

    N = 1000
    step = 0
    for step in range(N): # or while True:
        start = time.time()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy(tf_prev_state, ou_noise, add_noise = False) # policyde noise'i evaluate ederken 0 yap

        # Recieve state and reward from environment.        
        state, reward, done, info = env2.reward_calculation(action[0]) # reward is du-1 - du
        
        storage['pos']['x'].append(state[0])
        storage['pos']['y'].append(state[1])
        storage['pos']['z'].append(state[2])
        env2.render_calculate() # uncomment for animation

        print("{}th action".format(step))
        print("Goal k:{0}, phi:{1}, l:{2}".format(env2.k1,))
        print("Goal Position",state[3:6])
        print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state[0:3])) # for step_minus_euclidean_square
        print("Action: {0},  cable_lenghts {1}".format(action, env2.cab_lens))
        print("Reward is ", reward)
        print("--------------------------------------------------------------------------------")
        stop = time.time()
        env2.time += (stop - start)
        storage['error']['error_store'].append(math.sqrt(-1*reward)) # for step_minus_euclidean_square
        #storage['k']['k1'].append(env2.k1)
        #storage['k']['k2'].append(env2.k2)
        storage['cable length']['l1'].append(env2.cab_lens[0])
        storage['cable length']['l2'].append(env2.cab_lens[1])
        storage['cable length']['l3'].append(env2.cab_lens[2])
        storage['cable length']['l4'].append(env2.cab_lens[3])
        storage['cable length']['l5'].append(env2.cab_lens[4])
        storage['cable length']['l6'].append(env2.cab_lens[5])

        storage['error']['x'].append(abs(state[0]-state[3])) #x y z gx gy gz
        storage['error']['y'].append(abs(state[1]-state[4])) #0 1 2 3  4   5
        storage['error']['z'].append(abs(state[2]-state[5]))
        storage['reward']['value'].append(reward)
        # print(env.position_dic)
        
        # End this episode when `done` is True
        if done:
            # pass
            break
    storage['reward']['effectiveness'].append(step)
                           
time.sleep(1)
print(f'{env2.overshoot0} times robot tried to cross the task space')
print(f'{env2.overshoot1} times random goal was generated outside of the task space')
print(f'Simulation took {(env2.time)} seconds')
effectiveness_score = np.mean(storage['reward']['effectiveness'])
print(f'Average Effectiveness Score is {effectiveness_score}')

# %% Visualization of the results
#env2.visualization(storage['pos']['x'],storage['pos']['y'],storage['pos']['z'])
# plt.title(f"Initial Position is (x,y): ({initial_state[0]},{initial_state[1]}) & Target Position is (x,y): ({state[0]},{state[1]})",fontweight="bold")
#plt.xlabel("Position x [m]",fontsize=15)
#plt.ylabel("Position y [m]",fontsize=15)
#plt.show()
#env2.close()
# %%
# # uncomment below for animation 
# ani = env.render()
# video = ani.to_html5_video()
