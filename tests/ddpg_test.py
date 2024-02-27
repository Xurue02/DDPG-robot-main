# %%
import sys
sys.path.append('../')
sys.path.append('../environment')
sys.path.append('../tests')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
from env import robot_env
from ddpg import OUActionNoise, policy
plt.style.use('../continuum_robot/plot.mplstyle')
from plotsfunc import *
from matplotlib import animation
# %matplotlib notebook
from IPython import display