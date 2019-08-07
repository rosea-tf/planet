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

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools
from planet.models import base
from planet.models import RSSM

from planet.tools import nested

class RSSM_FastSlow(base.Base): #inherits from RNN cell


  def __init__(
      self, state_size, belief_size, embed_size,
      future_rnn=False, mean_only=False, min_stddev=0.1, slow_timescale=4, slow_ppn=0.25, ignore_slow=False):
    self._state_size = state_size # dim of latent state?: 30
    self._belief_size = belief_size # h: both these come from model_size->size in configs.py: 200
    self._embed_size = embed_size
    self.future_rnn = future_rnn

    # divide total # params between this network (own_*_size), and the subsidiary, slower network
    self._slow_ppn = slow_ppn
    self._slow_state_size = int(state_size * slow_ppn)
    self._slow_belief_size = int(belief_size * slow_ppn)
    self._slow_timescale = slow_timescale
    self._fast_state_size = state_size - (self._slow_state_size if not ignore_slow else 0)
    self._fast_belief_size = belief_size - (self._slow_belief_size if not ignore_slow else 0)
    self._ignore_slow = ignore_slow

    self._slow_rnn = RSSM(self._slow_state_size, self._slow_belief_size, embed_size,
    future_rnn, mean_only, min_stddev)
    self._fast_rnn = RSSM(self._fast_state_size, self._fast_belief_size, embed_size,
    future_rnn, mean_only, min_stddev)

    self._slow_count_prior = tf.Variable(0, trainable=False, dtype=tf.int32)
    self._slow_count_postr = tf.Variable(0, trainable=False, dtype=tf.int32)

    #make _transition_tpl, _posterior_tpl for variable sharing
    super(RSSM_FastSlow, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {
        'mean': self._state_size, #so mean is a ~30D vector...
        'stddev': self._state_size,
        'sample': self._state_size,
        'belief': self._belief_size, #observation space??
        'rnn_state': self._belief_size,
    }

  def dist_from_state(self, state, mask=None):
    """Extract the latent distribution from a prior or posterior state."""
    if mask is not None:
      stddev = tools.mask(state['stddev'], mask, value=1)
    else:
      stddev = state['stddev']
    dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
    return dist

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return tf.concat([state['sample'], state['belief']], -1)

  def divergence_from_states(self, lhs, rhs, mask):
    """Compute the divergence measure between two states."""
    """ADR NB: state is a tuple of ['mean','stddev', ...]"""
    lhs = self.dist_from_state(lhs, mask)
    rhs = self.dist_from_state(rhs, mask)
    return tools.mask(tfd.kl_divergence(lhs, rhs), mask)

  def _fastslow_merge(self, prev_state, prev_action, obs, slow_fn, fast_fn, slow_counter):
    # they have variable shapes (either 30 or 200), so...
    slow_prev_state = nested.map(lambda tensor: tensor[..., :int(int(tensor.shape[-1]) * self._slow_ppn)], prev_state)
    fast_prev_state = nested.map(lambda tensor: tensor[..., int(int(tensor.shape[-1]) * self._slow_ppn):], prev_state)
    
    #time check
    slow_next_state = tf.cond(tf.equal(slow_counter, 0), lambda: slow_fn(slow_prev_state, prev_action, obs), lambda: slow_prev_state)
    
    # slow_counter_print = tf.Print(slow_counter, [slow_counter], message="slow_count: ")

    slow_counter_op = tf.assign(slow_counter, tf.mod(tf.add(slow_counter, 1), self._slow_timescale))
    
    fast_next_state = fast_fn(fast_prev_state, prev_action, obs)

    with tf.control_dependencies([slow_counter_op]):
    # with tf.control_dependencies([slow_counter_op, slow_counter_print]):
      next_state = {k: tf.concat([slow_next_state[k], fast_next_state[k]], -1) for k in slow_next_state.keys()}
    
    return next_state
  
  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute prior next state by applying the transition dynamics."""
    
    if self._ignore_slow:
      return self._fast_rnn._transition_tpl(prev_state, prev_action, zero_obs)

    return self._fastslow_merge(prev_state, prev_action, zero_obs, self._slow_rnn._transition_tpl, self._fast_rnn._transition_tpl, self._slow_count_prior)

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""

    if self._ignore_slow:
      return self._fast_rnn._posterior_tpl(prev_state, prev_action, obs)

    return self._fastslow_merge(prev_state, prev_action, obs, self._slow_rnn._posterior_tpl, self._fast_rnn._posterior_tpl, self._slow_count_postr)
