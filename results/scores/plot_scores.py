#%%
import matplotlib.pyplot as plt
import json
import sys
import numpy as np
import pickle
from itertools import chain 

wd = 'results/scores'

# pick up scores
input_json = f'{wd}/scalars.json'

randscore = {}
with open(input_json, 'r') as f:
  score_dict = json.load(f)

score_dict['qb_t0_hl_pc'] = score_dict['qb_t0_hl_pc_x20']
# score_dict['bk_tp_hl_pc_fixd'] = score_dict['bk4_tp_hl_pc_fixd']
# score_dict['bk4_t0_hl_pc'] = score_dict['bk_t0_hl_pc']

print("Score Keys: ", score_dict.keys())

input_hist = f'{wd}/histograms.pkl'

with open(input_hist, 'rb') as f:
  hist_dict = pickle.load(f)

print("Hist Keys: ", hist_dict.keys())

#%% EXPLORATION AND DISCRETISATION
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(12, 4), sharex=True, sharey=True)

ax = ax1
ax.set_title('Exploration in Discrete Cartpole Balance')
for ex in (10, 20, 30):

  data = score_dict['dc_t0_h0_pc_x{}'.format(ex)]['test_score']

  ax.plot(data, label='$\epsilon={}$'.format(ex / 100))

ax.set_xlabel('Epochs')
ax.set_ylabel('Test score')
ax.set_xlim(0, 60)
ax.legend()

ax = ax2
ax.set_title('Discretisation in Cartpole Balance')

data = score_dict['dc_t0_h0_pc_x20']['test_score']
ax.plot(data, label='$n = 5$'.format(ex / 100))

data = score_dict['dcf_t0_h0_pc']['test_score']
ax.plot(data, label='$n = 9$'.format(ex / 100))

ax.set_xlabel('Epochs')
ax.set_xlim(0, 60)
ax.legend()

ax = ax3
ax.set_title('Discretisation and Rewards in Cartpole Swingup')

data = score_dict['du_t0_h0_pc']['test_score']
ax.plot(data, label='$n = 5$, Dense'.format(ex / 100))

data = score_dict['duf_t0_h0_pc']['test_score']
ax.plot(data, label='$n = 9$, Dense'.format(ex / 100))

data = score_dict['dusf_t0_h0_pc']['test_score']
ax.plot(data, label='$n = 9$, Sparse'.format(ex / 100))

ax.set_xlabel('Epochs')
ax.set_xlim(0, 60)
# ax.set_ylabel('Test score')
ax.legend()

fig.tight_layout()
fig.savefig(f'{wd}/expl-discrtn.png', dpi=150)

#%% ## HP TESTS (cheetah, cartpole)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(1, 3, 1)
ax = ax1  #*************
ax.set_title('Cheetah Run')

for code, fullname in zip(['p0', 'pc', 'p0_xos', 'pc_wos'], [
    'Original HP (with OS)', 'Updated HP (no OS)', 'Original HP (no OS)',
    'Updated HP (with OS)'
]):

  data = score_dict['ch_t0_h0_{}'.format(code)]['test_score']

  ax.plot(data, label=fullname)

ax.set_xlim(0, 60)
ax.set_xlabel('Epochs')
ax.set_ylabel('Test score')

ax2 = fig.add_subplot(1, 3, 2, sharex=ax1)
ax = ax2  #*************
ax.set_title('Continuous-Action Cartpole')

for code, fullname in zip(['p0', 'pc', 'p0_xos', 'pc_wos'], [
    'Original HP (with OS)', 'Updated HP (no OS)', 'Original HP (no OS)',
    'Updated HP (with OS)'
]):

  data = score_dict['cc_t0_h0_{}'.format(code)]['test_score']

  ax.plot(data, label=fullname)

ax.set_xlim(0, 60)
ax.set_xlabel('Epochs')

ax3 = fig.add_subplot(1, 3, 3, sharey=ax2)
ax = ax3  #*************
ax.set_title('Discrete-Action Cartpole')

for code, fullname in zip(['p0', 'pc', 'p0_xos', 'pc_wos'], [
    'Original HP (with OS)', 'Updated HP (no OS)', 'Original HP (no OS)',
    'Updated HP (with OS)'
]):

  data = score_dict['dc_t0_h0_{}'.format(code)]['test_score']

  ax.plot(data, label=fullname)

ax.set_xlim(0, 60)
ax.set_xlabel('Epochs')
# ax.set_ylabel('Test score')

ax.legend(*ax.get_legend_handles_labels(), loc='best')


# fig.legend(*ax2.get_legend_handles_labels(), loc='upper right')
fig.tight_layout()
fig.savefig(f'{wd}/test-byhp.png', dpi=150)

#%% ## MODEL TESTS
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)

for left in axs[:, 0]:
  left.set_ylabel('Test Score')

for bottom in axs[-1, :]:
  bottom.set_xlabel('Epochs')

for ax, scorekey, fullname, randkey in zip(
  axs.ravel(),
  ['ch_{}_h0_pc', 'du_{}_h0_pc', 'bk4_{}_hl_pc','qb_{}_hl_pc'],
  ['Cheetah Run', 'Discrete Cartpole Swingup', 'Atari Breakout', 'Atari Qbert'],
  [None, 'Cartpole', 'Breakout', 'Qbert']
  ):

  ax.set_title(fullname)
  ax.set_xlim(0, 60)

  for line_code, line_name in zip(
      # ['t0', 'tm', 'tc', 'ta'],
      # ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Time-Agnostic Loss']):
      ['t0', 'tm', 'tc', 'tp', 'ta'],
      ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Clockwork (w/ priors)', 'Time-Agnostic Loss']):
    
    try:
      data = score_dict[scorekey.format(line_code)]['test_score']
      ax.plot(data, label=line_name)
    except:
      try:
        data = score_dict[scorekey.format(line_code).replace('bk4', 'bk')]['test_score']
        ax.plot(data, label=line_name)
      except:
        continue
  
  # pick up scores from random agents, if needed
  if randkey:
    with open('{}/randreturn-{}-v0.json'.format(wd, randkey), 'r') as f:
      scores = json.load(f)

    randscore = np.percentile(scores, 99)

    ax.axhline(randscore, label='Random (99 pctile)', color='black', ls='--')


fig.legend(*axs[1,1].get_legend_handles_labels(), loc='right')
fig.tight_layout()
fig.subplots_adjust(left=0.167, right=0.8, wspace=0.33)
fig.savefig(f'{wd}/test-bymodel.png', dpi=150)



#%% ## MODEL LOSS
fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex='col', sharey='none')

axs[0, 0].set_ylabel('Image Loss')
axs[1, 0].set_ylabel('Reward Loss')

for bottom in axs[-1, :]:
  bottom.set_xlabel('Epochs')

for axpair, scorekey, fullname in zip(
  axs.T,
  ['ch_{}_h0_pc', 'du_{}_h0_pc', 'bk4_{}_hl_pc'],
  ['Cheetah Run', 'Discrete Cartpole Swingup', 'Atari Breakout'],
  ):

  axpair[0].set_title(fullname, pad=20)
  axpair[0].set_xlim(0, 60)
  axpair[0].ticklabel_format(useOffset=False)

  for line_code, line_name in zip(
      # ['t0', 'tm', 'tc', 'ta'],
      # ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Time-Agnostic Loss']):
      ['t0', 'tm', 'tc', 'tp', 'ta'],
      ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Clockwork (w/ priors)', 'Time-Agnostic Loss']):
    try:
      data = score_dict[scorekey.format(line_code)]['image_loss']
      axpair[0].plot(data, label=line_name)
    except:
      continue

    try:
      data = score_dict[scorekey.format(line_code)]['reward_loss']
      axpair[1].plot(data, label=line_name)
    except:
      continue
  

fig.legend(*axs[0,0].get_legend_handles_labels(), bbox_to_anchor=(0.98, 0.9))
#  loc='upper right')
fig.tight_layout()
# fig.subplots_adjust(left=0.167, right=0.8, wspace=0.33)
fig.savefig(f'{wd}/loss-bymodel.png', dpi=150)





#%% RANDOM EPIS
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey='row')
# top row: image loss
# second row: reward loss
# left: by hp
# right: by time model

axs[0, 0].set_ylabel('Image Loss')
axs[1, 0].set_ylabel('Reward Loss')

for bottom in axs[-1, :]:
  bottom.set_xlabel('Epochs')

# from left to right
c = 0
for axpair, scorekey, crnns, frnns, heading in zip(
  axs.T,
  ['bk4_t0_hl_{}_fixd', 'bk4_{}_hl_pc_fixd'],
  [
      ['p0', 'p0_rwt', 'pc'],
      ['t0', 'tm', 'tc', 'tp', 'ta'],
  ],
  [
      ['Original HP', 'Original HP + weight on reward loss', 'Updated HP'],
      ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Clockwork (w/ priors)', 'Time-Agnostic Loss'],
  ],
  ['Hyperparameter Comparison', 'Time Model Comparison'],
  ):

  axpair[0].set_title(heading, pad=20)
  axpair[0].set_xlim(0, 60)
  axpair[0].ticklabel_format(useOffset=False)

  for line_code, line_name in zip(crnns, frnns):

    try:
      data = score_dict[scorekey.format(line_code)]['image_loss']
      axpair[0].plot(data, label=line_name)
    except:
      continue

    try:
      data = score_dict[scorekey.format(line_code)]['reward_loss']
      axpair[1].plot(data, label=line_name)
    except:
      continue

  axpair[0].legend()
  axpair[1].legend()


# fig.legend(*axs[-1].get_legend_handles_labels(), loc='right')
fig.tight_layout()
fig.subplots_adjust(left=0.167, right=0.833, wspace=0.33) #total 0.33...

fig.savefig(f'{wd}/loss-random.png', dpi=150)

# %% CEM histograms

fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=False, sharex=True)

for line_code, line_name, ax in zip(
    ['t0', 'tm', 'tc', 'ta'],
    ['Standard RSSM', 'Moving Avg', 'Clockwork', 'Time-Agnostic Loss'], axs):
  data = hist_dict['bk_{}_hl_pc'.format(line_code)]['cem_returns']
  h = ax.hist(data, label=line_name, bins=range(9))
  ax.get_yaxis().set_visible(False)
  ax.set_title(line_name)

fig.tight_layout()
fig.savefig(f'{wd}/cem-reward-histogram.png', dpi=150)


#%% RANDOM EPIS - ALTERNATE TRAIN/TEST
fig, axs = plt.subplots(2, 4, figsize=(12, 7), sharex=True, sharey='row')
# top row: image loss
# second row: reward loss
# left: by hp
# right: by time model

axs[0, 0].set_ylabel('Image Loss')
axs[1, 0].set_ylabel('Reward Loss')

for bottom in axs[-1, :]:
  bottom.set_xlabel('Epochs')

# from left to right
c = 0
for b, (ax_4s, scorekey, line_codes, line_names, heading) in enumerate(zip(
  np.split(axs, 2, axis=1),
  ['bk4_t0_hl_{}_fixd', 'bk4_{}_hl_pc_fixd'],
  [
      ['p0', 'p0_rwt', 'pc'],
      ['tm', 'tc', 'tp'],
  ],
  [
      ['Original HP', 'Orig. HP (upweight reward loss)', 'Updated HP'],
      ['Moving Avg', 'Clockwork', 'Clockwork (w/ priors)'],
  ],
  ['By Hyperparam. ({})', 'By Model ({})'],
  )):

  ax_4s[0, 0].set_title(heading.format('Train'), pad=20)
  ax_4s[0, 1].set_title(heading.format('Test'), pad=20)

  for axrow in ax_4s:
    axrow[0].set_xlim(0, 60)
    axrow[0].ticklabel_format(useOffset=False)

  for l, (line_code, line_name) in enumerate(zip(line_codes, line_names)):
    
    c=f'C{b* 3 + l}'

    try:
      data = score_dict[scorekey.format(line_code)]['image_loss']
      ax_4s[0,0].plot(data, label=line_name, c=c)

      data = score_dict[scorekey.format(line_code)]['reward_loss']
      ax_4s[1,0].plot(data, label=line_name, c=c)

      data = score_dict[scorekey.format(line_code)]['image_loss_te']
      ax_4s[0,1].plot(data, label=line_name, c=c)

      data = score_dict[scorekey.format(line_code)]['reward_loss_te']
      ax_4s[1,1].plot(data, label=line_name, c=c)

    except:
      continue

  # axpair[0].legend()
  # axpair[1].legend()

(h1, l1), (h2, l2) =[
  axs[0,0].get_legend_handles_labels(), 
  axs[0,2].get_legend_handles_labels(),
  ]


# fig.legend(h1+h2, l1+l2, loc='lower center', ncol=6)
fig.legend(h1, l1, bbox_to_anchor=[0.275, 0.0], loc='lower center', ncol=6)

fig.legend(h2, l2, bbox_to_anchor=[0.75, 0.0], loc='lower center', ncol=6)

fig.tight_layout()
fig.subplots_adjust(bottom=0.125) #total 0.33...

fig.savefig(f'{wd}/loss-random.png', dpi=150)