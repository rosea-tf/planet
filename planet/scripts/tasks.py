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

import collections
import functools
import os

import numpy as np

from planet import control
from planet import networks
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance')
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_balance_da(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance', discretise=[[-1.0],[-0.25],[0.0],[0.25],[1.0]])
  return Task('cartpole_balance_da', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup')
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin')
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run')
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 6)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch')
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk')
  return Task('walker_walk', env_ctor, max_length, state_components)


def humanoid_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = [
      'reward', 'com_velocity', 'extremities', 'head_height', 'joint_angles',
      'torso_vertical', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'humanoid', 'walk')
  return Task('humanoid_walk', env_ctor, max_length, state_components)


def gym_cheetah(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'state']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'HalfCheetah-v3')
  return Task('gym_cheetah', env_ctor, max_length, state_components)


def gym_racecar(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'CarRacing-v0', obs_is_image=True)
  return Task('gym_racing', env_ctor, max_length, state_components)


def gym_breakout(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'Breakout-v0', obs_is_image=True)
  return Task('gym_breakout', env_ctor, max_length, state_components)


def gym_freeway(config, params):
  action_repeat = params.get('action_repeat', 3)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'Freeway-v0', obs_is_image=True)
  return Task('gym_freeway', env_ctor, max_length, state_components)


def _dm_control_env(action_repeat, max_length, domain, task, discretise=None):
  from dm_control import suite
  env = control.wrappers.DeepMindWrapper(suite.load(domain, task), (64, 64))
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)

  # if we are discretising:
    # a wrapper that converts list of fixed actions in original space [[x1, y1], [x2, y2], [x3, y3]]
    # into an artificial n-dim space [a, b, c]
    # which then gets selected with combinatorial CEM and maxed, as before.

  if discretise:
    env = control.wrappers.Discretizer(env, action_set=discretise)
    env = control.wrappers.DiscreteActionWrap(env)

  return env


def _gym_env(action_repeat, min_length, max_length, name, obs_is_image=False, explicit_frameskip=True):
  import gym

  if not explicit_frameskip:
    env = gym.make(name)
    env = control.wrappers.ActionRepeat(env, action_repeat)
  else:
    env = gym.make(name, frameskip=action_repeat)
  
  ### ADR: additions to cope with discrete action spaces
  if isinstance(env.action_space, gym.spaces.Box):
    env = control.wrappers.NormalizeActions(env)
  elif isinstance(env.action_space, gym.spaces.Discrete):
    env = control.wrappers.DiscreteActionWrap(env) # ADR new thing
    # pass
  else:
    raise NotImplementedError("Unsupported action space '{}.'".format(env.action_space))
  ### end additions

  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length)
  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env
