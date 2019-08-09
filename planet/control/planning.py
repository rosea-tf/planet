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

from planet.control import discounted_return
from planet import tools


def cross_entropy_method(
    cell, objective_fn, state, obs_shape, action_shape, horizon,
    amount=1000, topk=100, iterations=10, discount=0.99,
    min_action=-1, max_action=1,
    discrete_action=False, tap_cell=None, tap_state=None): #ADR
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)

  # ADR
  if tap_cell is not None:
    initial_tap_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), tap_state)

  
  # pick up the standard batch size as used in 'sample', 'belief', etc
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  obs = tf.zeros((extended_batch, horizon) + obs_shape)
  length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  def iteration(mean_and_stddev, _):
    mean, stddev = mean_and_stddev
    # Sample action proposals from belief.
    
    if not discrete_action:
        normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
        action = normal * stddev[:, None] + mean[:, None] #insert new dim at 2nd pos
        action = tf.clip_by_value(action, min_action, max_action)
    else:
        # note, action shape should be 1D here!
        # sample from a categorical dist
        probs_flat = tf.reshape(mean, [-1, action_shape[0]]) #[oh, a]=probs
        choice_flat = tf.random.categorical(probs_flat, amount) #[oh, m]=choice
        action_flat = tf.one_hot(choice_flat, depth=action_shape[0], axis=-1) #[oh, m, a]={0,1}
        action = tf.reshape(action_flat, [original_batch, horizon, amount, action_shape[0]]) #[o, h, m, a]
        action = tf.transpose(action, [0, 2, 1, 3]) #[o, m, h, a]

    # Evaluate proposal actions.
    action = tf.reshape(
        action, (extended_batch, horizon) + action_shape)
    (_, state), _ = tf.nn.dynamic_rnn(
        cell, (0 * obs, action, use_obs), initial_state=initial_state)
    reward = objective_fn(state)

    if tap_cell is not None:
      # maybe do the objective function hack here...
      (_, tap_state), _ = tf.nn.dynamic_rnn(tap_cell, (0 * obs, action, use_obs), initial_state=initial_tap_state)

      # add an extra dimension because it's what div_from_states wants
      state_ = tools.nested.map(lambda tensor: tensor[:, :, None], state)
      tap_state_ = tools.nested.map(lambda tensor: tensor[:, :, None], tap_state)
      blank_mask = tf.ones(tools.shape(state_['mean'])[0:3], dtype=tf.dtypes.bool)

      divergence = tap_cell.divergence_from_states(state_, tap_state_, blank_mask)
      print("divergence set up")
      #print divergence TODO
      divergence = tf.squeeze(divergence) #get rid of 1 at end
      divergence = tf.Print(divergence, [divergence])

      reward = tf.subtract(reward, divergence)

    return_ = discounted_return.discounted_return(
        reward, length, discount)[:, 0]
    return_ = tf.reshape(return_, (original_batch, amount))
    # Re-fit belief to the best ones.
    _, indices = tf.nn.top_k(return_, topk, sorted=False)
    indices += tf.range(original_batch)[:, None] * amount
    best_actions = tf.gather(action, indices) #[o,k,h,a]
    
    if not discrete_action:
        mean, variance = tf.nn.moments(best_actions, 1) #[o,h,a]
        stddev = tf.sqrt(variance + 1e-6)
    else:
        # count the number of times each action chosen
        mean = tf.reduce_mean(best_actions, axis=1) #[o,h,a]
        mean = tf.math.log(mean + 1e-6)#[o,h,a]

    # mean = tf.Print(mean, [mean[0, 0]], summarize=8, message="Mean: ")

    return mean, stddev

  mean = tf.zeros((original_batch, horizon) + action_shape)
  # mean = tf.Print(mean, [mean[0, 0]], summarize=8, message="NEW PLAN: ")
  # initialise a gaussian with mean zero
  # in discrete case, these will be logprobs for a gen. bernoulli
  
  # this will only have effect in the gaussian/continuous case
  stddev = tf.ones((original_batch, horizon) + action_shape)
  
  mean, stddev = tf.scan(
      iteration, tf.range(iterations), (mean, stddev), back_prop=False) #[i,o,h,a]
  
  mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations: [o,h,a]
      
  return mean 
