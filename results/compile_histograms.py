import tensorflow as tf
import time
import csv
import sys
import os
import collections
import json
import numpy as np
import pickle

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False
# TF version < 1.1.0
if (not eventAccumulatorImported):
  try:
    from tensorflow.python.summary import event_accumulator
    eventAccumulatorImported = True
  except ImportError:
    eventAccumulatorImported = False
# TF version = 1.1.0
if (not eventAccumulatorImported):
  try:
    from tensorflow.tensorboard.backend.event_processing import event_accumulator
    eventAccumulatorImported = True
  except ImportError:
    eventAccumulatorImported = False
# TF version >= 1.3.0
if (not eventAccumulatorImported):
  try:
    from tensorboard.backend.event_processing import event_accumulator
    eventAccumulatorImported = True
  except ImportError:
    eventAccumulatorImported = False
# TF version = Unknown
if (not eventAccumulatorImported):
  raise ImportError('Could not locate and import Tensorflow event accumulator.')

# summariesDefault = ['scalars','histograms','images','audio','compressedHistograms']

class Timer(object):
  # Source: https://stackoverflow.com/a/5849861
  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    if self.name:
      print('[%s]' % self.name)
      print('Elapsed: %s' % (time.time() - self.tstart))

root_folder = sys.argv[1]
output_file = sys.argv[2]

def get_hists(log_folder, graphkey):

  with Timer():
    ea = event_accumulator.EventAccumulator(log_folder,
      size_guidance={
          event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
          # event_accumulator.IMAGES: 0,
          # event_accumulator.AUDIO: 0,
          # event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 0,
    })

  with Timer():
    ea.Reload() # loads events from file

  # event_acc.Reload()
  # tags = event_acc.Tags()
  
  histograms = ea.Histograms(graphkey)

  result = np.array([np.repeat(np.array(h.histogram_value.bucket_limit), np.array(h.histogram_value.bucket).astype(np.int)) for h in histograms])
  
  # print("duhh", result)

  return result

results = {}

# [subdir, scalar key]
hist_dict = {
  'cem_returns': ['test', 'graph/summaries/simulation/should_simulate_gym_breakout/summary-gym_breakout-cem-60/return_hist'],
}

for loc, dirs, files in os.walk(root_folder):
  print('Looking at :', loc)
  for dir in dirs:
    if dir == '00001': # we have a run folder here
      print('Found an 00001 at :', loc)
      
      # initialise results sub-dictionary
      loc_results = {}

      for hist, [subdir, graphkey] in hist_dict.items():
        path = os.path.join(loc, dir, subdir)
        loc_results[hist] = get_hists(path, graphkey)
        # try:
        #   loc_results[hist] = get_hists(path, graphkey)
        #   print(graphkey, ' fetched from :', path)
        # except:
        #   print("Couldn't get ", graphkey, " from ", path)
        #   loc_results[hist] = [0]
      
      results[os.path.basename(loc)] = loc_results

with open(output_file, 'wb') as outfile:
    pickle.dump(results, outfile)


