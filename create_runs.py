import collections
import os
import pathlib
from string import Template


with open("create_runs_template.sh", "r") as template_file:
  template = Template(template_file.read())



default_params = ['max_steps: 3e6']

Setting = collections.namedtuple('Setting', 'code, params, desc')

set_expl = [
    Setting('x10', 'exploration_noises: [0.1]', 'Low epsilon exploration'),
    Setting('x20', 'exploration_noises: [0.2]', 'Med epsilon exploration'),
    Setting('x30', 'exploration_noises: [0.3]', 'High epsilon exploration'),
]

set_null = [
    Setting('', '', ''),
]

set_hyp = [
    Setting('p0', '', 'HPs from Paper'),
    Setting('p0_rwt', 'overshooting_reward_scale: 1e5, reward_scale: 1e5', 'HPs from Paper (1e5 Reward Wt)'),
    Setting(
        'pc',
        'future_rnn: true, free_nats: 3.0, overshooting: 0, global_divergence_scale: 0.0, overshooting_reward_scale: 0.0',
        'Camera Ready HPs'),
    Setting(
        'p0_xos',
        'overshooting: 0, overshooting_reward_scale: 0.0',
        'HPs from Paper (Minus OS)'),
    Setting(
        'pc_wos',
        'future_rnn: true, free_nats: 3.0, global_divergence_scale: 0.0',
        'Camera Ready HPs (Plus OS)'),
    Setting(
        'pc_rwt',
        'future_rnn: true, free_nats: 3.0, overshooting: 0, global_divergence_scale: 0.0, overshooting_reward_scale: 0.0, reward_scale: 1e5',
        'Camera Ready HPs (1e5 Reward Wt)'),
]


set_fixd = [
    Setting('', '', 'Standard'),
    Setting('fixd', 'collect_every: 999999999, num_seed_episodes: 1000', 'Fixed Random Episodes'),
]

set_env = [
    Setting('ch', 'tasks: [cheetah_run]', 'Cheetah'),
    Setting('cc', 'tasks: [cartpole_balance]', 'Continuous cartpole'),
    Setting('dc', 'tasks: [cartpole_balance_da], discrete_action: true', 'Discretised cartpole'),
    Setting('du', 'tasks: [cartpole_swingup_da], discrete_action: true', 'Discretised cartpole-swingup'),
    Setting('dcf', 'tasks: [cartpole_balance_daf], discrete_action: true', 'Discretised cartpole (fine)'),
    Setting('duf', 'tasks: [cartpole_swingup_daf], discrete_action: true', 'Discretised cartpole-swingup (fine)'),
    Setting('dusf', 'tasks: [cartpole_swingupsparse_daf], discrete_action: true', 'Discretised cartpole-swingup (fine)'),
    Setting('bk4', 'tasks: [gym_breakout], discrete_action: true', 'Atari Breakout'),
    Setting('qb', 'tasks: [gym_qbert], discrete_action: true', 'Atari Qbert'),
]

set_rnn = [
    Setting('t0', '', 'Vanilla RSSM'),
    Setting('tm', 'model: rssm_ma', 'Moving-Average RSSM'),
    # Setting('ta', 'tap_cell: rssm, batch_shape: [4, 30]', 'Time-Agnostic Loss'),
    Setting('ta', 'tap_cell: rssm, batch_shape: [4, 30], model_size: [100], state_size: [15]', 'Time-Agnostic Loss'),
    Setting('tc', 'model_size: [100, 100], state_size: [15, 15], cw_tau: [1, 4]', 'Clockwork RNN'),
    Setting('tp', 'model_size: [100, 100], state_size: [15, 15], cw_tau: [1, 4], cell_as_prior: 0.1', 'Clockwork-Prior RNN'),
]

hrz_length = {'bk4': 60, 'fw': 30, 'qb': 30}

set_hrz = [
    # Setting('h0', '', 'Standard Horizon'),
    Setting('hl', lambda x: f'collect_horizons: [{x}], summary_horizons: [{x}]', 'Long Horizon'),
]
#TODO
i = 0
for env in set_env:
  for rnn in set_rnn:
    # if rnn.code != 't0': 
      # if env.code == 'ch':
      #   #TODO: train cheetah only on best-performing rnn
      #   continue
      # if env.code in ['cc', 'dc']:
        # don't bother on cartpole
        # continue
    for hrz in set_hrz:
      if hrz.code == 'hl':
        if env.code not in hrz_length.keys():
          continue
        hrz = Setting(hrz.code, hrz.params(hrz_length[env.code]), hrz.desc)
      for hyp in set_hyp:
        # _set_expl = set_expl if env.code in ['dc','qb'] else set_null
        # for expl in _set_expl:
        for fixd in set_fixd:

          # cfg = [env, rnn, hrz, hyp, expl]
          cfg = [env, rnn, hrz, hyp, fixd]
          label = '_'.join([c.code for c in cfg if c.code != ''])

          # marker for random datasets
          # label += '_fixd'

          params = ', '.join(default_params + [c.params for c in cfg if c.params != ''])
          
          i += 1
          print(i, ': --> '.join([label, params]))
          
          outpath = f'runs/{label}'

          pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

          # generate script file at this location
          """
          assume structure is:
          ~
            planet
              planet
              runs
                curr ** scripts target here
                todo
                done        
          # runs
          # --> doing --> 
          """
          with open(os.path.join(outpath, f'{label}.sh'), 'w', newline='\n') as out_file:
            out_file.write(template.safe_substitute({'LABEL': label, 'PARAMS': params}))
        