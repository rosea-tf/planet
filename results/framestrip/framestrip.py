#%%
import os
os.getcwd()
#%%
from PIL import Image
import numpy as np

wd = 'results/framestrip'


#%%

def strip(file, D, start, step, qty):

  arr = np.array(Image.open(file))
  
  nh, nw = [d // D for d in arr.shape]

  assert(step * qty <= (nh * nw) - start) #enough?

  def frame(i):
    ih = i // nw
    iw = i % nw
    return arr[ih * D:(ih + 1) * D, iw * D:(iw + 1) * D]

  strip = np.empty((D, D * qty))
  
  for i in range(qty):
    strip[:, i * D:(i + 1) * D] = frame(start + (i * step))

  return strip

#2nd half of 3rd row looks good

fnames = ['bk_t0_hl_pc_train_data','bk_t0_hl_pc_train_openloop','bk_t0_hl_pc_train_closedloop']

titles = ['Environment','Closed loop','Open loop']

startsteps = [ #start, step
  (100, 5), # row 3
  (300, 5), # row 7
  (350, 5), # row 8
] 

ims = [[strip(f"{wd}/{name}.png", 64, start, step, 10) for name in names] for (start, step) in startsteps]

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

for i_ep, (start, step) in enumerate(startsteps):

  fig, axs = plt.subplots(3, 1, figsize=(12,4), sharex=True, sharey=True)

  fig.suptitle(f'Sequence {i_ep + 1}')

  # it_axs = iter(axs)

  for ax, fname, title in zip(axs, fnames, titles):
    
    # ax = next(it_axs)
    im = strip(f"{wd}/{fname}.png", 64, start, step, 10)

    # ax.axis('off')
    ax.tick_params(size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_facecolor("none")
    for pos in ["left", "right", "top", "bottom"]:
        ax.spines[pos].set_visible(False)

    ax.set_ylabel(title)
    ax.imshow(im, cmap='gist_heat_r')

  fig.tight_layout()
  fig.subplots_adjust(top=.9)
  fig.savefig(f'{wd}/framestrip_{i_ep}.png', dpi=150)


# %%
