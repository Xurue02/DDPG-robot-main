# %%
import sys
sys.path.append('../../../../')
from plotsfunc.utils import *
# %%
avg_reward_list = load_pickle_file('avg_reward_list')
ep_reward_list = load_pickle_file('ep_reward_list')

# %%
## Plotting graph
reward_visualization(ep_reward_list, avg_reward_list)
plt.savefig('reward_visual.png')
plt.show()
# %%
# Plotting graph log scale
reward_log10_visualization(ep_reward_list, avg_reward_list)
plt.savefig('reward_visual_log.png')
plt.show()
# %%