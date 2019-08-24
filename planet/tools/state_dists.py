from planet import tools
from tensorflow_probability import distributions as tfd

def dist_from_state(state, mask=None):
  """Extract the latent distribution from a prior or posterior state."""
  if mask is not None:
    stddev = tools.mask(state['stddev'], mask, value=1)
  else:
    stddev = state['stddev']
  dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
  return dist


def divergence_from_states(lhs, rhs, mask):
  """Compute the divergence measure between two states."""
  """ADR NB: state is a tuple of ['mean','stddev', ...]"""
  lhs = dist_from_state(lhs, mask)
  rhs = dist_from_state(rhs, mask)
  return tools.mask(tfd.kl_divergence(lhs, rhs), mask)