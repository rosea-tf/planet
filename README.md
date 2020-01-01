# PlaNet with discrete-action adaptation and augmented transition model

Alex Rose

The following new parameters can be passed as `--params {...}` in addition to the syntax detailed in the original README below.

| Experiment | Parameters |
| :--------- | :--------- |
| Updated (aka Camera Ready) HPs | `future_rnn: true, free_nats: 3.0, global_divergence_scale: 0.0, overshooting_reward_scale: 0.0` |
| Use discrete actions (also requires a discrete-action environment) | `discrete_action: true` |
| Change default epsilon-greedy parameter to _x_ | `exploration_noises: [x]` |
| Moving-Average RSSM, optionally with 'slow' proportion _x_ and speed parameter _y_ | `model: rssm_ma, ma_ppn: x, ma_alpha: y` |
| Clockwork RNN with deterministic component sizes _w, x, ..._; stochastic component sizes _y, z, ..._ and update frequency _m, n, ..._ | `model_size: [w, x], state_size: [y, z], cw_tau: [m, n]` |
| (for Clockwork RNN) use slow cells as priors with strength _x_ for fast cells (in this case, component sizes must be equal) | `cell_as_prior: x` |
| Time-Agnostic Losses | `tap_cell: rssm` |
| Track found CEM returns in Tensorboard | `summarise_plan_returns: true` |
| Warm starts for the CEM planner | `warm_start: true` |
| Save model state as .npz files | `collect_latents: true` |


The following new discrete environments (defined in `tasks.py`) can also be passed as `--params {...}`.

| Environment | Parameters |
| :--------- | :--------- |
| Cartpole Balance | `tasks: [cartpole_balance_da]` |
| Cartpole Swingup | `tasks: [cartpole_swingup_da]` |
| Cartpole Balance (Fine control) | `tasks: [cartpole_balance_daf]` |
| Cartpole Swingup (Fine control) | `tasks: [cartpole_swingup_daf]` |
| Cartpole Swingup (Fine control, Sparse Reward) | `tasks: [cartpole_balance_daf]` |
| Atari Breakout | `tasks: [gym_breakout]` |
| Atari Qbert | `tasks: [gym_qbert]` |
| Atari Freeway | `tasks: [gym_freeway]` |


Scores from completed runs under a parent folder can be compiled into JSON format with:

``python results/scores/compile_scores.py path/to/parent/folder/ path/to/output/scalars.json``

``python results/scores/compile_histograms.py path/to/parent/folder/ path/to/output/histograms.json``

Original `README.MD` follows...
_______________________
_______________________

# Deep Planning Network

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson

![PlaNet policies and predictions](https://imgur.com/UeeQIfo.gif)

This project provides the open source implementation of the PlaNet agent
introduced in [Learning Latent Dynamics for Planning from Pixels][paper].
PlaNet is a purely model-based reinforcement learning algorithm that solves
control tasks from images by efficient planning in a learned latent space.
PlaNet competes with top model-free methods in terms of final performance and
training time while using substantially less interaction with the environment.

If you find this open source release useful, please reference in your paper:

```
@article{hafner2018planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  journal={arXiv preprint arXiv:1811.04551},
  year={2018}
}
```

## Method

![PlaNet model diagram](https://i.imgur.com/fpvrAqw.png)

PlaNet models the world as a compact sequence of hidden states. For planning,
we first encode the history of past images into the current state. From there,
we efficiently predict future rewards for multiple action sequences in latent
space. We execute the first action of the best sequence found and replan after
observing the next image.

Find more information:

- [Google AI Blog post][blog]
- [Animated paper][website]
- [Paper as PDF][paper]

[blog]: https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html
[paper]: https://danijar.com/publications/2019-planet.pdf
[website]: https://planetrl.github.io/

## Instructions

To train an agent, install the dependencies and then run:

```sh
python3 -m planet.scripts.train  \
    --logdir /path/to/logdir \
    --config default \
    --params '{tasks: [cheetah_run]}'
```

The code prints `nan` as the score for iterations during which no summaries
were computed.

The available tasks are listed in `scripts/tasks.py`. The default parameters
can be found in `scripts/configs.py`. To run the experiments from our
paper, pass the following parameters to `--params {...}` in addition to the
list of tasks:

| Experiment | Parameters |
| :--------- | :--------- |
| PlaNet | No additional parameters. |
| No overshooting | `overshooting: 0` |
| Random dataset | `collect_every: 999999999, num_seed_episodes: 1000` |
| Purely deterministic | `overshooting: 0, mean_only: True, divergence_scale: 0.0, global_divergence_scale: 0.0` |
| Purely stochastic | `model: ssm` |
| One agent all tasks | `collect_every: 30000` |

Please note that the agent has seen some improvements so the results may be a
bit different now.

## Modifications

These are good places to start when modifying the code:

| Directory | Description |
| :-------- | :---------- |
| `scripts/configs.py` | Add new parameters or change defaults. |
| `scripts/tasks.py` | Add or modify environments. |
| `models` | Add or modify latent transition models. |
| `networks` | Add or modify encoder and  decoder networks. |

Tips for development:

- You can set `--config debug` to reduce the episode length, batch size, and
  collect data more freqnently. This helps to quickly reach all parts of the
  code.
- You can use `--num_runs 1000 --resume_runs False` to automatically start new
  runs in sub directories of the logdir every time to execute the script.
- Environments live in separate processes by default. Some environments work
  better when separated into threads instead by specifying `--params
  '{isolate_envs: thread}'`.

## Dependencies

The code was tested under Ubuntu 18 and uses these packages:

- tensorflow-gpu==1.13.1
- tensorflow_probability==0.6.0
- dm_control (`egl` [rendering option][dmc-rendering] recommended)
- gym
- scikit-image
- scipy
- ruamel.yaml
- matplotlib

[dmc-rendering]: https://github.com/deepmind/dm_control#rendering

Disclaimer: This is not an official Google product.
