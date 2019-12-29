try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

Perf = namedtuple('Perf', 'frameskip horizon amount rewards')

fig = plt.figure(figsize=(10,4))

for i, envt in enumerate(['Breakout-v0', 'Qbert-v0']):


  with open(f'realcem/realcem_{envt}.pkl', 'rb') as file:
    perfs = pickle.load(file)
    perfs = [p for p in perfs if p.frameskip >= 2]
    print(envt, set([p.horizon for p in perfs]))

  ax = fig.add_subplot(1, 2, i + 1, projection = "3d")
  ax.set_title(envt.replace('-v0',''), pad=20)
  ax.set_xlabel("Frameskip")
  ax.set_ylabel("Horizon length") 
  ax.set_zlabel("Best return found")
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

  # ax.set_xlim3d(0,4)
  # ax.set_ylim3d(0,4) 

  #horizon
  xpos = [p.frameskip for p in perfs]
  ypos = [p.horizon for p in perfs]
  zpos = [0 for p in perfs]

  dx = np.ones_like(xpos)
  dy = np.ones_like(ypos)

  amounts = [p.amount for p in perfs]

  dz = [np.random.random(21) for _ in range(4)]  # the heights of the 4 bar sets

  # _zpos = zpos   # the starting zpos for each bar
  colors = {500: 'r', 1000: 'b', 1500: 'g', 2000: 'g'}

  for p in perfs:
    reward = p.rewards[-1].sum()
    ax.bar3d(p.frameskip + (min(int(p.amount / 500), 3) - 1) * 0.2 - 0.3, 0 if p.horizon == 12 else p.horizon - 6, 0, 0.2, 12, np.npv(0.00, p.rewards[-1]), color=colors[p.amount], alpha=0.5)
    # _zpos += dz[i]    # add the height of each bar to know where to start the next


  plt.gca().invert_xaxis()

dummies = [plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.5) for col in ['r','b','g']]

fig.legend(
  dummies,
  ['500','1000','1500'], title='CEM samples per iteration'
)
fig.tight_layout()
plt.savefig(f'realcem/horizon_test', dpi=200)