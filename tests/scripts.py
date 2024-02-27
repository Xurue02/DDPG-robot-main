
import sys
sys.path.append('../')
sys.path.append('../environment')
import tensorflow as tf
from tensorflow.keras import layers
from env import robot_env

env = robot_env()
print(env.observation_space.shape[0])
num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states));
inputs = layers.Input(shape=(num_states,))
print(inputs)