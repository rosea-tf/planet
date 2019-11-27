import tensorflow as tf
import time
import csv
import sys
import os
import collections
import json

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

def get_scores(log_folder, graphkey):

  with Timer():
    ea = event_accumulator.EventAccumulator(log_folder,
      size_guidance={
          # event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
          # event_accumulator.IMAGES: 0,
          # event_accumulator.AUDIO: 0,
          event_accumulator.SCALARS: 0,
          # event_accumulator.HISTOGRAMS: 0,
    })

  with Timer():
    ea.Reload() # loads events from file
  
  return [e.value for e in ea.Scalars(graphkey)]
  # [e.value for e in ea.Scalars('trainer/graph/phase_test/cond_2/trainer/test/score')]

results = {}

# [subdir, scalar key]
scalar_dict = {
  'test_score': ['test', 'trainer/graph/phase_test/cond_2/trainer/test/score'],
  'image_loss': ['train', 'graph/summaries/general/zero_step_losses/image'], 
  'reward_loss': ['train', 'graph/summaries/general/zero_step_losses/reward']
}

for loc, dirs, files in os.walk(root_folder):
  print('Looking at :', loc)
  for dir in dirs:
    if dir == '00001': # we have a run folder here
      print('Found an 00001 at :', loc)
      
      # initialise results sub-dictionary
      loc_results = {}

      for scalar, [subdir, graphkey] in scalar_dict.items():
        path = os.path.join(loc, dir, subdir)
        try:
          loc_results[scalar] = get_scores(path, graphkey)
          print(graphkey, ' fetched from :', path)
        except:
          print("Couldn't get ", graphkey, " from ", path)
          loc_results[scalar] = [0]
      
      results[os.path.basename(loc)] = loc_results

with open(output_file, 'w') as outfile:
    json.dump(results, outfile)

# if ('scalars' in summaries):
# 	print(' ')
# 	csvFileName =  os.path.join(outputFolder,'scalars.csv')
# 	print('Exporting scalars to csv-file...')
# 	print('   CSV-path: ' + csvFileName)
# 	scalarTags = tags['scalars']
# 	with Timer():
# 		with open(csvFileName,'w') as csvfile:
# 			logWriter = csv.writer(csvfile, delimiter=',')

# 			# Write headers to columns
# 			headers = ['wall_time','step']
# 			for s in scalarTags:
# 				headers.append(s)
# 			logWriter.writerow(headers)
	
# 			vals = ea.Scalars(scalarTags[0])
# 			for i in range(len(vals)):
# 				v = vals[i]
# 				data = [v.wall_time, v.step]
# 				for s in scalarTags:
# 					scalarTag = ea.Scalars(s)
# 					S = scalarTag[i]
# 					data.append(S.value)
# 				logWriter.writerow(data)


