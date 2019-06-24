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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools


def encoder(obs, dumbnet=False, diff_frame=False):
  """Extract deterministic features from an observation."""

  # if not diff_frame:
  # use full set of image observations
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list()) #50x64x64x3 result
  # else:
  # chop off first observation in each sequence (which we don't have a previous image for)
  # hidden = tf.reshape(obs['image'][:, 1:, ...], [-1] + obs['image'].shape[2:].as_list()) #50x64x64x3 result

  if not dumbnet:
    kwargs = dict(strides=2, activation=tf.nn.relu)
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs) #31x31
    hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs) #14x14
    hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs) #6x6
    hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs) #2x2
  else:
    #do a one-step convolution
    hidden = tf.layers.conv2d(hidden, 256, 4, strides=32, activation=tf.nn.relu) #2x2

  hidden = tf.layers.flatten(hidden)
  assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()

  if diff_frame:
    # calculate difference image
    to_subtract = tf.concat(
        [obs['image'][:, 0:1, ...], obs['image'][:, :-1, ...]], axis=1)
    diff = tf.subtract(obs['image'], to_subtract)

    # merge batch and seq dimensions, as before
    diff = tf.reshape(diff, [-1] + diff.shape[2:].as_list())

    # do the same convolution, but with half as many filters
    kwargs = dict(strides=2, activation=tf.nn.relu)
    diff = tf.layers.conv2d(diff, 16, 4, **kwargs) #31x31
    diff = tf.layers.conv2d(diff, 32, 4, **kwargs) #14x14
    diff = tf.layers.conv2d(diff, 64, 4, **kwargs) #6x6
    diff = tf.layers.conv2d(diff, 128, 4, **kwargs) #2x2

    #flatten
    diff = tf.layers.flatten(diff)
    assert diff.shape[1:].as_list() == [512], diff.shape.as_list() #TODO

    #append it to the flattened hidden layer

    hidden = tf.concat([hidden, diff], 1)
    assert hidden.shape[1:].as_list() == [1536], hidden.shape.as_list()

  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
  np.prod(hidden.shape[1:].as_list())])

  # batch_seq_dim = tools.shape(obs['image'])[:2]

  # if diff_frame: batch_seq_dim[1] -= 1

  # hidden = tf.reshape(hidden, batch_seq_dim + hidden.shape[1:].as_list())

  return hidden


def decoder(state, data_shape, dumbnet=False):
  """Compute the data distribution of an observation from its state."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.layers.dense(state, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])

  if not dumbnet:
    hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 3, 6, strides=2)

  else:
    hidden = tf.layers.conv2d_transpose(hidden, 2, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 2, 5, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 2, 6, **kwargs)
    hidden = tf.layers.conv2d_transpose(hidden, 3, 6, strides=2)

  mean = hidden
  assert mean.shape[1:].as_list() == [64, 64, 3], mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tfd.Normal(mean, 1.0)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
