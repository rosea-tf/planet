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


class RSSM(base.Base): #inherits from RNN cell
  r"""Deterministic and stochastic state model.

  The stochastic latent is computed from the hidden state at the same time
  step. If an observation is present, the posterior latent is compute from both
  the hidden state and the observation.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  [h]->[h]  [h]->[h]
      ^ |       ^ :
     /  v      /  v
  (s)  (s)  (s)  (s)
                  ^
                  :
                 (o)
  """ # is this diagram what "temporal prior" refers to?

  """
  
  tf.RNN cells (such as this one) must implement:
    (output, next_state) = call(input, state)

  RSSM inherits from base:
    def call(self, inputs, prev_state):
    ...    
    prior = self._transition_tpl(prev_state, prev_action, zero_obs)
    ...
    posterior = tf.cond(
        use_obs,
        lambda: self._posterior(prev_state, prev_action, obs),
        lambda: prior)
    return (prior, posterior), posterior

    so  output = (pri, post) (prior used for lat. overshooting)
        next_state = post
  """


  def __init__(
      self, state_size, belief_size, embed_size,
      future_rnn=False, mean_only=False, min_stddev=0.1):
    self._state_size = state_size # dim of latent state?: 30
    self._belief_size = belief_size # h: both these come from model_size->size in configs.py: 200
    self._embed_size = embed_size
    self.future_rnn = future_rnn
    self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size) #num_units (i.e. output size)=200
    self._kwargs = dict(units=self._embed_size, activation=tf.nn.relu)
    self._mean_only = mean_only
    self._min_stddev = min_stddev
    super(RSSM, self).__init__(
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

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return tf.concat([state['sample'], state['belief']], -1)

  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute prior next state by applying the transition dynamics."""
    # a_1, h_1 -> h_2 in Fig 2c
    inputs = tf.concat([prev_state['sample'], prev_action], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    # returns output, hidden_state. Per GRU diagram and TF source code, these are actually the same! bx200

    if self.future_rnn:
      hidden = belief
    
    hidden = tf.layers.dense(hidden, **self._kwargs) #??? RNN output used in posterior
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev #??? not a min operation?
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': belief,
        'rnn_state': rnn_state,
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    #obs dim 1024: 
    prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
    inputs = tf.concat([prior['belief'], obs], -1) #200 + 1024 = 224
    hidden = tf.layers.dense(inputs, **self._kwargs) #200 again
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': prior['belief'],
        'rnn_state': prior['rnn_state'],
    }
