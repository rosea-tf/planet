# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.tools import nested
from planet.tools import shape as tools_shape


class MPCAgent(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._batch_env = batch_env
    self._step = step  # Trainer step, not environment step.
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._cell = config.cell
    self._tap_cell = config.tap_cell
    state = self._cell.zero_state(len(batch_env), tf.float32)
    var_like = lambda x: tf.get_local_variable(
        x.name.split(':')[0].replace('/', '_') + '_var',
        shape=x.shape,
        initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
    self._state = nested.map(var_like, state)

    if self._tap_cell is not None:
      tap_state = self._tap_cell.zero_state(len(batch_env), tf.float32)
      self._tap_state = nested.map(var_like, tap_state)


    self._prev_action = tf.get_local_variable(
        'prev_action_var', shape=self._batch_env.action.shape,
        initializer=lambda *_, **__: tf.zeros_like(self._batch_env.action),
        use_resource=True)
    # ADR - new - for CEM warm starts
    if self._config.warm_start:
      orig_shape = tools_shape(self._batch_env.action)
      warm_start_shape = [
        orig_shape[0],
          self._config['planner'].keywords['horizon']
      ] + orig_shape[1:]
      self._warm_start = tf.get_local_variable(
        'warm_start', shape=warm_start_shape,
        initializer=lambda *_, **__: tf.zeros(warm_start_shape, dtype=tf.float32),
        use_resource=True)
    else:
      self._warm_start = None


  def begin_episode(self, agent_indices):
    state = nested.map(
        lambda tensor: tf.gather(tensor, agent_indices),
        self._state)
    reset_state = nested.map(
        lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
        self._state, state, flatten=True)
    reset_prev_action = self._prev_action.assign(
        tf.zeros_like(self._prev_action))

    # ADR also back to zero at start of each episode
    if self._config._warm_start:
      reset_warm_start = self._warm_start.assign(tf.zeros_like(self._warm_start))
    else:
      reset_warm_start = tf.no_op()

    with tf.control_dependencies(reset_state + (reset_prev_action, reset_warm_start)):
      return tf.constant('')

  def perform(self, agent_indices, observ):
    observ = self._config.preprocess_fn(observ)
    embedded = self._config.encoder({'image': observ[:, None]})[:, 0]
    state = nested.map(
        lambda tensor: tf.gather(tensor, agent_indices),
        self._state)

    if self._tap_cell is not None:
      tap_state = nested.map(
          lambda tensor: tf.gather(tensor, agent_indices),
          self._tap_state)
    else:
      tap_state = None
    

    prev_action = self._prev_action + 0
    with tf.control_dependencies([prev_action]):
      use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
      _, state = self._cell((embedded, prev_action, use_obs), state)

      if self._tap_cell is not None:
        _, tap_state = self._tap_cell((embedded, prev_action, use_obs), tap_state)
    
    # note, self._config is actually an agent_config object, 
    # not the usual config object that would contain the discrete_action flag
    # so we do this hacky thing.
    discrete_action = self._config.planner.keywords['discrete_action']

    # get the means (or log probs, in the discrete case)
    mean, single, plan_returns = self._config.planner(
        cell=self._cell,
        objective_fn=self._config.objective, state=state,
        obs_shape=embedded.shape[1:].as_list(),
        action_shape=prev_action.shape[1:].as_list(), 
        warm_start=self._warm_start,
        tap_cell=self._tap_cell, tap_state=tap_state,
        ) #[o,h,a]=actvalue, [o,m]=r
      # PARTIALS
      # amount=params.get('cem_amount', 1000),
      # topk=params.get('cem_topk', 100),
      # iterations=params.get('cem_iterations', 10),
      # horizon=horizon,
      # discrete_action=params.get('discrete_action'))
      
    # keep only the first action of the n-step sequence: we will replan over again on the next step
    if not discrete_action:
      action = mean[:, 0] #[o,a]
    else:
      #in discrete case, choose best single trajectory, rather than the mean
      action = single[:, 0]

    if self._config.exploration:
      scale = self._config.exploration.scale
      if self._config.exploration.schedule:
        scale *= self._config.exploration.schedule(self._step)

      if not discrete_action:
        action = tfd.Normal(action, scale).sample()
      else:
        # for each batch item, determine whether we explore (e-greedy strategy)
        expl_sample = tf.random.uniform(action.shape[0:1])
        expl_do = tf.math.greater(scale, expl_sample)
        # action probs get replaced with random probs
        action = tf.where(expl_do, tf.random.uniform(action.shape), action)

    if not discrete_action:
      action = tf.clip_by_value(action, -1, 1)
    else:
      # pick highest prob action
      action = tf.contrib.seq2seq.hardmax(action)

    remember_action = self._prev_action.assign(action)
    remember_state = nested.map(
        lambda var, val: tf.scatter_update(var, agent_indices, val),
        self._state, state, flatten=True)

    #ADR - new
    if self._config.warm_start:
      # cut off first action slot, add zero-means at end
      remember_warm = self._warm_start.assign(
          tf.concat([mean[:, 1:], tf.zeros_like(mean[:, 0:1])], axis=1))
    else:
      remember_warm = tf.no_op()

    if self._config.summarise_plan_returns:
      agent_extras = {'plan_returns_begin': plan_returns[0], 'plan_returns_end': plan_returns[-1]} #[i,o,m]
    else:
      agent_extras = dict()

    with tf.control_dependencies(remember_state + (remember_action, remember_warm)):
      return tf.identity(action), tf.constant(''), agent_extras #ADR - new

  def experience(self, agent_indices, *experience):
    return tf.constant('')

  def end_episode(self, agent_indices):
    return tf.constant('')
