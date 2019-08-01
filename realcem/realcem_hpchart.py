#%%
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
from collections import namedtuple

Perf = namedtuple('Perf', 'frameskip horizon amount rewards')

with open('realcem/perfs.pkl', 'rb') as file:
  perfs_1 = pickle.load(file)
with open('realcem/perfs_2.pkl', 'rb') as file:
  perfs_2 = pickle.load(file)

perfs = [p for p in perfs_1 if p.amount == 1000] + [p for p in perfs_2]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

ax.set_xlabel("frameskip")
ax.set_ylabel("horizon") 
ax.set_zlabel("return")


ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


# ax.set_xlim3d(0,4)
# ax.set_ylim3d(0,4) 

#horizon

# x = np.arange(7) + 1
# y = np.arange(3) + 1
# xpos, ypos = [a.ravel() for a in np.meshgrid(x, y)]
# xpos = [1,2,3,1,2,3,1,2,3]
# xpos = ['a','b','c','a','b','c','a','b','c',]
#frameskip
# xpos = [1,1,1,2,2,2,3,3,3]
xpos = [p.frameskip for p in perfs]
ypos = [p.horizon for p in perfs]
zpos = [0 for p in perfs]
#samples
# zpos = np.zeros(21)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)

amounts = [p.amount for p in perfs]

dz = [np.random.random(21) for i in range(4)]  # the heights of the 4 bar sets

# _zpos = zpos   # the starting zpos for each bar
colors = {500: 'r', 1000: 'b', 1500: 'g'}

for p in perfs:
  reward = p.rewards[-1].sum()
  ax.bar3d(p.frameskip + (int(p.amount / 500) - 1) * 0.2 - 0.3, 0 if p.horizon == 12 else p.horizon - 6, 0, 0.2, 12, np.npv(0.00, p.rewards[-1]), color=colors[p.amount], alpha=0.5)
  # _zpos += dz[i]    # add the height of each bar to know where to start the next


plt.gca().invert_xaxis()
plt.show()

#%%
perfs[0].rewards.shape