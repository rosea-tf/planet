import numpy as np
import gym
import tensorflow as tf
# from planet.control.planning import cross_entropy_method
# from planet.control import siscounted_return
from collections import namedtuple
import json

try:
   import cPickle as pickle
except:
   import pickle

def cross_entropy_method(
    # cell, objective_fn, state, obs_shape, action_shape,
    env,
    horizon,
    amount=1000,
    topk=100,
    iterations=10,
    discount=0.99,
    min_action=-1,
    max_action=1,
    discrete_action=True):  #ADR
  """modified, a bit, for this script"""

  obs_shape, action_shape = env.observation_space.shape, [
      env.action_space.n
  ] if discrete_action else env.action_space.shape

  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  # original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  # original_batch = 1
  # extended_batch = 1 * amount

  # initial_state = tools.nested.map(lambda tensor: tf.tile(
  # tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)

  # pick up the standard batch size as used in 'sample', 'belief', etc
  # extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  # use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  # obs = tf.zeros((extended_batch, horizon) + obs_shape)
  # length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  def iteration(mean, stddev):
    
    # Sample action proposals from belief.

    if not discrete_action:
      normal = tf.random_normal((original_batch, amount, horizon) +
                                action_shape)
      action = normal * stddev[:, None] + mean[:,
                                               None]  #insert new dim at 2nd pos
      action = tf.clip_by_value(action, min_action, max_action)
    else:
      # note, action shape should be 1D here!
      # sample from a categorical dist
      # probs_flat = tf.reshape(mean, [horizon, -1])  # tf. cat requires: batch size, num classes
      # probs = np.exp(mean)
      # probs_cum = np.cumsum(probs, axis=1)
      # probs_cum = probs_cum / probs_cum[:, -1:] #normalise
      # r = np.random.rand(probs.shape[0], amount)
      choices = tf.random.categorical(mean, amount).numpy().transpose()  #[oh, m]=choice
      # choice = (probs_cum[..., None] < r).sum(axis=1)

      actions = tf.one_hot(choices, depth=action_shape[0], axis=-1).numpy()  #[oh, m, a]={0,1}
      # action = tf.reshape(
          # action_flat,
          # [original_batch, horizon, amount, action_shape[0]])  #[o, h, m, a]
      # action = tf.transpose(action, [0, 2, 1, 3])  #[o, m, h, a]

    # Evaluate proposal actions.
    # action = tf.reshape(action, (extended_batch, horizon) + action_shape)

    snapshot = env.ale.cloneState()
    returns = np.empty([amount])
    reward_over_samples = np.zeros([amount, horizon])

    for m in range(amount):
      # choices = np.argmax(action[m], axis=1)
      transitions = [env.step(c) for c in choices[m]]
      # transitions = [transition() for transition in transitions]
      observs, rewards, dones, infos = zip(*transitions)
      returns[m] = np.npv(1 - discount, rewards)
      reward_over_samples[m] = rewards

      env.ale.restoreState(snapshot)

    # (_, state), _ = tf.nn.dynamic_rnn(
    #     cell, (0 * obs, action, use_obs), initial_state=initial_state)
    # reward = objective_fn(state)
    # return_ = discounted_return.discounted_return(
    # reward, length, discount)[:, 0]
    # return_ = tf.reshape(returns, (original_batch, amount))
    # Re-fit belief to the best ones.
    # return_topk, indices = tf.nn.top_k(return_, topk, sorted=False)
    # indices += tf.range(original_batch)[:, None] * amount
    indices = np.argpartition(returns, -topk)[-topk:]
    # best_actions = tf.gather(action, indices)  #[o,k,h,a]
    best_actions = actions[indices]  #[o,k,h,a]

    reward_topk = np.mean(reward_over_samples[indices], axis=0)

    if not discrete_action:
      mean, variance = tf.nn.moments(best_actions, 1)  #[o,h,a]
      stddev = np.sqrt(variance + 1e-6)
    else:
      # count the number of times each action chosen
      mean = np.mean(best_actions, axis=0)  #[o,h,a]
      mean = np.log(mean + 1e-6)  #[o,h,a]

    return mean, stddev, reward_topk

  mean = np.ones((horizon, action_shape[0]))

  # this will only have effect in the gaussian/continuous case
  stddev = np.ones((horizon, action_shape[0]))

  reward_over_iters = np.zeros([iterations, horizon])
  # return_ = tf.constant(0, dtype=tf.float64)

  for i in range(iterations):
    mean, stddev, reward_over_iters[i] = iteration(mean, stddev)

  # mean, stddev, return_ = tf.scan(
      # iteration, tf.range(iterations), (mean, stddev, return_),
      # back_prop=False)  #[i,o,h,a]

  mean, stddev, = mean[-1], stddev[-1]  # Select belief at last iterations: [o,h,a]

  return reward_over_iters


tf.enable_eager_execution()

Perf = namedtuple('Perf', 'frameskip horizon amount rewards')


for envt in ['Freeway-v0', 'Qbert-v0']: #, Breakout-v0 already tested
  perfs = []
  for frameskip in (1, 2, 3, 4, 5):
    # in all cases: default is to sample uniformly from {2,3,4}
    env = gym.make(envt, frameskip=frameskip)
    for horizon in (12, 24, 36, 48, 60, 72, 84):
      for amount in (500, 1000, 2000):
        env.reset()

        # one random step to get it going
        env.step(np.random.randint(env.action_space.n))

        # env.render()
        print("Horizon {}, amount {}".format(horizon, amount))
        rewards = cross_entropy_method(env, horizon)
        print("reward max", rewards.max())
        # perfs['{}/{}/{}'.format(frameskip, horizon, amount)] = return_.numpy()
        perfs.append(Perf(frameskip, horizon, amount, rewards))
        # with open('perfs.json', 'w') as outfile:
          # json.dump(perfs, outfile)

        with open(f'realcem_{envt}.pkl', 'wb') as outfile:
          pickle.dump(perfs, outfile)

  env.close()