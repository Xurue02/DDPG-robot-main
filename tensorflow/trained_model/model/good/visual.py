# %%
import sys
sys.path.append('../../../../')
from plotsfunc.utils import *
# %%
avg_reward_list = load_pickle_file('avg_reward_list')
ep_reward_list = load_pickle_file('ep_reward_list')

# %%
## Plotting graph
'''
reward_visualization(ep_reward_list, avg_reward_list)
plt.savefig('reward_visual.png')
plt.show()
# %%
# Plotting graph log scale
reward_log10_visualization(ep_reward_list, avg_reward_list)
plt.savefig('reward_visual_log.png')
plt.show()
'''
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")


    
    # Episodes versus Rewards
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
plt.savefig('reward_visual.png')
# %%