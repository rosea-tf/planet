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


class SSM(base.Base):
  """Gaussian state space model.

  Implements the transition function and encoder using feed forward networks.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  (s)->(s)  (s)->(s)
                  ^
                  :
                 (o)
  """

  def __init__(self, state_size, belief_size, embed_size, future_rnn, mean_only, min_stddev):
    # belief_size and future_rnn are not used. added only for a constructor consistent with other cell types
    # removed defaults for future_rnn, mean_only, min_stddev - these should always be specified via the .partial constructor set up in configs.py

    self._state_size = state_size
    self._embed_size = embed_size
    self._mean_only = mean_only
    self._min_stddev = min_stddev
    super(SSM, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))
    self._kwargs = dict(units=self._embed_size, activation=tf.nn.relu)

  @property
  def state_size(self):
    return {
        'mean': self._state_size,
        'stddev': self._state_size,
        'sample': self._state_size,
    }

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return state['sample']

  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute prior next state by applying the transition dynamics."""
    inputs = tf.concat([prev_state['sample'], prev_action], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
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
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
    inputs = tf.concat([prior['mean'], prior['stddev'], obs], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
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
    }
