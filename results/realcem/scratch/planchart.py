#%%
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np

with open('realcem/pack.pkl', 'rb') as file:
    pack = pickle.load(file)
    
mean_over_iters, stddev_over_iters, best_path_over_iters, reward_over_iters, reward_topk_over_iters = pack


#%%

import gym
env = gym.make('Breakout-v0')
env.get_action_meanings()
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

#%%
import matplotlib.pyplot as plt
mean_over_iters.shape

plt.clf()
for i in range(10):
  x = np.arange(60)
  pr_lt = np.exp(mean_over_iters[i, :, 3])
  pr_rt = np.exp(mean_over_iters[i, :, 2])
  cl = '<--C{}'.format(i)
  cr = '>:C{}'.format(i)
  c = 'C{}'.format(i)

  # plt.plot(x, pr_lt, cl)
  # plt.plot(x, pr_rt, cr)
  plt.plot(np.cumsum(pr_rt - pr_lt), x, c, label=i)
  # plt.plot(x, pr_rt - pr_lt, c, label=i)
  plt.legend()

plt.show()

#%%

best_path_over_iters[-1]


#%%

reward_topk_over_iters.shape

for i in range(0,1):
  plt.plot(reward_topk_over_iters[i].mean(axis=0), label=i)
  plt.legend()

plt.show()