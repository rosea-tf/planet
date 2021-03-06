# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import functools

import tensorflow as tf
import tensorflow_probability as tfp
from math import ceil

from planet import tools
from planet.training import define_summaries, utility
from planet.tools.overshooting import _merge_dims
from planet.tools.state_dists import dist_from_state, divergence_from_states
from planet.tools.tf_repeat import tf_repeat


def define_model(data, trainer, config):
  tf.logging.info('Build TensorFlow compute graph.')
  dependencies = []
  step = trainer.step
  global_step = trainer.global_step
  phase = trainer.phase
  should_summarize = trainer.log

  # Preprocess data.
  with tf.device('/cpu:0'):
    if config.dynamic_action_noise:
      data['action'] += tf.random_normal(
          tf.shape(data['action']), 0.0, config.dynamic_action_noise)
    prev_action = tf.concat([0 * data['action'][:, :1], data['action'][:, :-1]],
                            1)
    obs = data.copy()
    del obs['length']

  # Instantiate network blocks.
  cells = []
  for i, _cell in enumerate(config.cells):
    with tf.variable_scope('cell_{}'.format(i)):
      cells.append(_cell())  #RSSM, etc
  del i, _cell
  
  kwargs = dict()

  if config.tap_cell is None:
    tap_cell = None
  else:
    tap_cell = config.tap_cell()

  #ADR - arguments for encoder
  kwargs['dumbnet'] = config.dumbnet

  encoder = tf.make_template(
      'encoder', config.encoder, create_scope_now_=True, **kwargs)
  heads = {}
  for key, head in config.heads.items():
    name = 'head_{}'.format(key)
    kwargs = dict(data_shape=obs[key].shape[2:].as_list())

    #ADR
    # if key == 'image':
    kwargs['dumbnet'] = config.dumbnet

    heads[key] = tf.make_template(name, head, create_scope_now_=True, **kwargs)

  # Embed observations and unroll model.
  #[5x10x1024] <- [..., 5x10x64x64x3, ..., ... (other elements in obs: ignored)]
  embedded = encoder(obs)  

  # Separate overshooting and zero step observations because computing
  # overshooting targets for images would be expensive.
  zero_step_obs = {}  # both of these are 5x10xd
  overshooting_obs = {}
  for key, value in obs.items():
    if config.zero_step_losses.get(key):
      zero_step_obs[key] = value
    if config.overshooting_losses.get(key):
      overshooting_obs[key] = value  #for now, they're identical
  assert config.overshooting <= config.batch_shape[1]

  max_os_length = config.overshooting + 1

  if len(config.cw_taus) == 1:
    # do it the classic way

    target, prior, posterior, mask = tools.overshooting(
      cells[0], overshooting_obs, embedded, prev_action, data['length'],
      config.overshooting + 1
    )  
    # prior is result of unrolling. posterior is what the unroll starts from.
  
  else:
  
    # CLOCKWORK MOD: each cell overshoots independently
    lstCells_lstOuts_dictTens = []
    
    for i, (_cell, _tau) in enumerate(zip(cells, config.cw_taus)):
      # calculate a sequence of inputs for each tau! (i.e. skip some for slow cells)
      _overshooting_obs = tools.nested.map(lambda tensor: tensor[:, ::_tau], overshooting_obs)
      _embedded = embedded[:, ::_tau]
      _prev_action = prev_action[:, ::_tau]
      _amount = ceil(max_os_length / _tau)

      with tf.variable_scope('cell_{}'.format(i)):
        _lstOutputs_dictTensors = tools.overshooting(
            cell=_cell, target=_overshooting_obs, embedded=_embedded, prev_action=_prev_action, length=data['length'],
            amount=_amount) 

      lstCells_lstOuts_dictTens.append(_lstOutputs_dictTensors) 

    del i, _cell, _tau
    # now need to "expand" slow-speed results

    # take these values straight from the first (tau=1) network
    target, mask = lstCells_lstOuts_dictTens[0][0], lstCells_lstOuts_dictTens[0][3]
    
    lstCells_prpo_dictTensx = [
      tools.nested.map(lambda tensor: 
        tf.concat(
          [
            tf_repeat(tensor[:, :, 0:1], [1, tau])[:, :config.batch_shape[1]],
            tf_repeat(tensor[:, :, 1:], [1, tau, tau])[:, :config.batch_shape[1], :max_os_length] #it may over-tile, so we cut it off
          ]
          , axis=2)
    , cellout[1:3])
    for cellout, tau in zip(lstCells_lstOuts_dictTens, config.cw_taus)]
    # [cell0:(target{p,r,v}, prior{b,m,r,s,s}, posterior..., mask), cell1:(target, prior, posterior, mask), ...]

    prpo_dictTens_tupCells = tools.nested.zip(*lstCells_prpo_dictTensx)
    # [target: {p: (cell0, cell1, ...), r:., ...}, prior: {b: (cell0, cell1, ...), ...}}]
    
    prior, posterior = [{k: tf.concat(v, axis=-1) for k, v in d.items()} for d in prpo_dictTens_tupCells]

  extra_tensors = None

  if config.collect_latents:
  # ADR
  # cut out overshooting: we want only posteriors from images
    extra_tensors = {
        'sample': posterior['sample'][:, :, 0, :],
        'belief': posterior['belief'][:, :, 0, :],
        'position': data['position'],
        'velocity': data['velocity']
    }

    # combine batch and sequence axes of all tensors
    extra_tensors = tools.nested.map(lambda tensor: _merge_dims(tensor, [0, 1]),
                                      extra_tensors)

  losses = []

  # Zero step losses
  # :1 = only first prediction. result=[5x10x1x30]: a horizontal row in fig3c
  _, zs_prior, zs_posterior, zs_mask = tools.nested.map(
      lambda tensor: tensor[:, :, :1], (target, prior, posterior, mask))
  zs_target = {
      key: value[:, :, None] for key, value in zero_step_obs.items()
  }  
  
  zero_step_losses = utility.compute_losses(
      loss_scales=config.zero_step_losses, 
      cells=cells,  # used for .divergence_from_states(), etc
      heads=heads,  # outputs other than prior, posterior
      step=step,
      target=zs_target,
      prior=zs_prior,
      posterior=zs_posterior,
      mask=zs_mask,
      free_nats=config.free_nats,
      debug=config.debug)
  losses += [
      loss * config.zero_step_losses[name]
      for name, loss in zero_step_losses.items()
  ]  #scale gets applied here
  if 'divergence' not in zero_step_losses:
    zero_step_losses['divergence'] = tf.zeros((), dtype=tf.float32)

  # Overshooting losses.
  if config.overshooting > 1:
    os_target, os_prior, os_posterior, os_mask = tools.nested.map(
        lambda tensor: tensor[:, :, 1:-1], (target, prior, posterior, mask))
    # everything AFTER step 0 in prediction/11 dim
    if config.stop_os_posterior_gradient:
      os_posterior = tools.nested.map(tf.stop_gradient, os_posterior)
    overshooting_losses = utility.compute_losses(
        config.overshooting_losses,  #doesn't include image
        cells,
        heads,
        step,
        os_target,
        os_prior,
        os_posterior,
        os_mask,
        config.free_nats,
        debug=config.debug)
    losses += [
        loss * config.overshooting_losses[name]
        for name, loss in overshooting_losses.items()
    ]
  else:
    overshooting_losses = {}
  if 'divergence' not in overshooting_losses:
    overshooting_losses['divergence'] = tf.zeros((), dtype=tf.float32)

  # ADR - time agnostic predictions
  if tap_cell is not None:
    # and then for a subgoal...
    # in planning step: obj fn = divergence between tap_cell-dist and cell-dist

    tap_losses = utility.compute_tap_losses(
      loss_scales=config.zero_step_losses, 
      cell=tap_cell, # same everything else, but use the separate network
      heads=heads,  
      step=step,
      target=zs_target,
      prior=zs_prior,
      posterior=zs_posterior,
      mask=zs_mask,
      free_nats=config.free_nats,
      debug=config.debug)

    losses += [
        loss * config.zero_step_losses[name] # we reuse the zs loss scales here... but should reward be down-weighted?
        for name, loss in tap_losses.items()
    ]  #scale gets applied here
      
    if 'divergence' not in tap_losses: 
      tap_losses['divergence'] = tf.zeros((), dtype=tf.float32) #why? oh well.
  
  
  # Workaround for TensorFlow deadlock bug.
  loss = sum(losses)
  train_loss = tf.cond(
      tf.equal(phase, 'train'), lambda: loss, lambda: 0 * tf.get_variable(
          'dummy_loss', (), tf.float32))
  train_summary = utility.apply_optimizers(train_loss, step, should_summarize,
                                           config.optimizers)
  # train_summary = tf.cond(
  #   tf.equal(phase, 'train'),
  #   lambda: utility.apply_optimizers(
  #     loss, step, should_summarize, config.optimizers),
  #   str, name='optimizers')

  # Active data collection.
  collect_summaries = []
  graph = tools.AttrDict(locals())
  with tf.variable_scope('collection'):
    should_collects = []
    for name, params in config.sim_collects.items():
      after, every = params.steps_after, params.steps_every
      should_collect = tf.logical_and(
          tf.equal(phase, 'train'),
          tools.schedule.binary(step, config.batch_shape[0], after, every))
      collect_summary, _ = tf.cond(  #planner comes in here
          should_collect,
          functools.partial(
              utility.simulate_episodes,
              config,
              params,
              graph,
              expensive_summaries=False,
              name=name),
          lambda: (tf.constant(''), tf.constant(0.0)),
          name='should_collect_' + params.task.name)
      should_collects.append(should_collect)
      collect_summaries.append(collect_summary)

  # Compute summaries.
  graph = tools.AttrDict(locals())
  with tf.control_dependencies(collect_summaries):
    summaries, score = tf.cond(
        should_summarize,
        lambda: define_summaries.define_summaries(graph, config),
        lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
        name='summaries')
  with tf.device('/cpu:0'):
    summaries = tf.summary.merge([summaries, train_summary] + collect_summaries)
    zs_entropy = (
        tf.reduce_sum(
            tools.mask(
                dist_from_state(zs_posterior, zs_mask).entropy(), zs_mask))
        / tf.reduce_sum(tf.to_float(zs_mask)))
    dependencies.append(
        utility.print_metrics((
            ('score', score),
            ('loss', loss),
            ('zs_entropy', zs_entropy),
            ('zs_divergence', zero_step_losses['divergence']),
        ), step, config.mean_metrics_every))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)

  # return score, summaries
  return score, summaries, extra_tensors
