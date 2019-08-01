import numpy as np
import gym
import tensorflow as tf
from collections import namedtuple
import json
import time

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

  def iteration(mean, stddev):

    # Sample action proposals from belief.

    if not discrete_action:
      normal = tf.random_normal((original_batch, amount, horizon) +
                                action_shape)
      action = normal * stddev[:, None] + mean[:,
                                               None]  #insert new dim at 2nd pos
      action = tf.clip_by_value(action, min_action, max_action)
    else:
      choices = tf.random.categorical(
          mean, amount).numpy().transpose()  #[m, h]=choice

      actions = tf.one_hot(
          choices, depth=action_shape[0], axis=-1).numpy()  #[m,h,a]={0,1}

    # snapshot = env.ale.cloneState()
    returns = np.empty([amount])
    reward_over_samples = np.zeros([amount, horizon])

    for m in range(amount):
      transitions = [env.step(c) for c in choices[m]]
      observs, rewards, dones, infos = zip(*transitions)
      # assert max(dones) == False
      returns[m] = np.npv(1 - discount, rewards)
      reward_over_samples[m] = rewards

      env.seed(0)
      env.reset()
      # env.ale.restoreState(snapshot)

    indices = np.argpartition(returns, -topk)[-topk:]  #[k]
    best_actions = actions[indices]  #[k,h,a]
    best_path = np.argmax(actions[np.argmax(returns)], axis=-1)  #[h]

    reward_topk = reward_over_samples[indices]  #[k, h]

    reward = np.mean(reward_topk, axis=0)  #[h]

    if not discrete_action:
      mean, variance = tf.nn.moments(best_actions, 1)  #[o,h,a]
      stddev = np.sqrt(variance + 1e-6)
    else:
      # count the number of times each action chosen
      mean = np.mean(best_actions, axis=0)  #[h,a]
      stddev = np.std(best_actions, axis=0)
      mean = np.log(mean + 1e-6)  #[h,a]

    return mean, stddev, best_path, reward, reward_topk

  mean = np.ones((horizon, action_shape[0]))

  # this will only have effect in the gaussian/continuous case
  stddev = np.ones((horizon, action_shape[0]))

  mean_over_iters = np.zeros([iterations, horizon, action_shape[0]])
  stddev_over_iters = np.zeros([iterations, horizon, action_shape[0]])
  reward_over_iters = np.zeros([iterations, horizon])
  reward_topk_over_iters = np.zeros([iterations, topk, horizon])
  best_path_over_iters = np.zeros([iterations, horizon])

  for i in range(iterations):
    mean_over_iters[i], stddev_over_iters[i], best_path_over_iters[
        i], reward_over_iters[i], reward_topk_over_iters[i] = iteration(
            mean, stddev)

  # mean, stddev, = mean[-1], stddev[-1]  # Select belief at last iterations: [o,h,a]

  return mean_over_iters, stddev_over_iters, best_path_over_iters, reward_over_iters, reward_topk_over_iters


tf.enable_eager_execution()

Perf = namedtuple('CEMPerformance', 'frameskip horizon amount rewards')
perfs = []

frameskip = 4
horizon = 60
amount = 1000

env = gym.make('Breakout-v0', frameskip=frameskip)
env.seed(0)
env.reset()
# env.render()

done = False
total = 0

while not done:
  pack = cross_entropy_method(env, horizon, discount=0.98)

  with open('realcem/pack.pkl', 'wb') as file:
    pickle.dump(pack, file)
  mean_over_iters, stddev_over_iters, best_path_over_iters, reward_over_iters, reward_topk_over_iters = pack

  for i, c in enumerate(best_path_over_iters[-1]):
    env.render()
    time.sleep(0.25)
    obs, reward, done, info = env.step(int(c))
    if done:
      break
    total += reward
    print("steP:{}, reward: {}".format(i, reward))
  break

# env.render()
# print("Horizon {}, amount {}".format(horizon, amount))
# rewards = cross_entropy_method(env, horizon)
# print("reward max", rewards.max())
# perfs.append(Perf(frameskip, horizon, amount, rewards))

# with open('perfs.pkl', 'wb') as outfile:
# pickle.dump(perfs, outfile)

env.close()

#%%
# with open('rewards.pkl', 'rb') as file:
#   rewards = pickle.load(file)

# rewards.shape

# rewards.sum(axis=1)

#%%