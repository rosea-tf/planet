import numpy as np
import gym
import tensorflow as tf
# from planet.control.planning import cross_entropy_method
# from planet.control import siscounted_return
from collections import namedtuple
import json

def cross_entropy_method(
    # cell, objective_fn, state, obs_shape, action_shape, 
    env,
    horizon,
    amount=1000, topk=100, iterations=10, discount=0.99,
    min_action=-1, max_action=1,
    discrete_action=True): #ADR

  """modified, a bit, for this script"""

  obs_shape, action_shape = env.observation_space.shape, [env.action_space.n] if discrete_action else env.action_space.shape

  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  # original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
  original_batch = 1
  extended_batch = 1 * amount

  # initial_state = tools.nested.map(lambda tensor: tf.tile(
      # tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
  
  # pick up the standard batch size as used in 'sample', 'belief', etc
  # extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  # use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  # obs = tf.zeros((extended_batch, horizon) + obs_shape)
  length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  def iteration(mean_and_stddev, _):
    mean, stddev, _ = mean_and_stddev
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

    # reward = tf.zeros_like(choice)

    snapshot = env.ale.cloneState()
    returns = np.empty([amount])
    
    for m in range(amount):
      choices = np.argmax(action[m], axis=1)    
      transitions = [env.step(c) for c in choices]
      # transitions = [transition() for transition in transitions]
      observs, rewards, dones, infos = zip(*transitions)
      returns[m] = np.npv(1 - discount, rewards)
      
      env.ale.restoreState(snapshot)

    x = 2
    
    # (_, state), _ = tf.nn.dynamic_rnn(
    #     cell, (0 * obs, action, use_obs), initial_state=initial_state)
    # reward = objective_fn(state)
    # return_ = discounted_return.discounted_return(
        # reward, length, discount)[:, 0]
    return_ = tf.reshape(returns, (original_batch, amount))
    # Re-fit belief to the best ones.
    return_topk, indices = tf.nn.top_k(return_, topk, sorted=False)
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

    # print ("Iteration: Best reward {}, avg all {} topk {}".format(
    #   tf.reduce_max(return_).numpy(),
    #   tf.reduce_mean(return_).numpy(),
    #   tf.reduce_mean(return_topk).numpy(),
    # ))
    return mean, stddev, tf.reduce_max(return_).numpy()

  mean = tf.zeros((original_batch, horizon) + action_shape)
  # mean = tf.Print(mean, [mean[0, 0]], summarize=8, message="NEW PLAN: ")
  # initialise a gaussian with mean zero
  # in discrete case, these will be logprobs for a gen. bernoulli
  
  # this will only have effect in the gaussian/continuous case
  stddev = tf.ones((original_batch, horizon) + action_shape)

  return_ = tf.constant(0, dtype=tf.float64)
  
  mean, stddev, return_ = tf.scan(
      iteration, tf.range(iterations), (mean, stddev, return_), back_prop=False) #[i,o,h,a]
  
  mean, stddev, return_ = mean[-1], stddev[-1], return_[-1]  # Select belief at last iterations: [o,h,a]
      
  return mean, return_


tf.enable_eager_execution()
env = gym.make('Breakout-v0')

# Perf = namedtuple('CEMPerformance', 'horizon amount return_')
perfs = {}
for horizon in (12, 24, 36, 48, 60, 72, 84):
  for amount in (500, 1000, 2000):
    env.reset()
    # env.render()
    print ("Horizon {}, amount {}".format(horizon, amount))
    action, return_ = cross_entropy_method(env, horizon)
    print("return", return_.numpy())
    perfs['{}/{}'.format(horizon, amount)] = return_.numpy()

    with open('perfs.json', 'w') as outfile:
      json.dump(perfs, outfile)
    
env.close()