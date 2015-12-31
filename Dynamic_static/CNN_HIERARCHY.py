# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================

import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES
import copy
import numpy as np

class CNN_H(object):
	"""docstring for CNN_H"""
	def __init__(self):
		self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
		subject_set = self.VARS.get_subject_set()

		print 'Creating ORIGINAL data set'
		convertion = self.VARS.CONVERTION_ORIGINAL
		self.data_set_ORIGINAL = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None)
		
		# print 'Creating STATIC DYNAMIC data set'
		# convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
		# self.data_set_SD = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None)


		# print 'Creating STATIC data set'
		# ''' Removes dynamic activities '''
		# remove_activities = self.VARS.REMOVE_DYNAMIC_ACTIVITIES
		# keep_activities = self.VARS.CONVERTION_STATIC
		# self.data_set_STATIC = input_data_window_large.read_data_sets_without_activity(subject_set, self.VARS.len_convertion_list(keep_activities), remove_activities, None, keep_activities)

		# print 'Creating DYNAMIC data set'
		# ''' Removes static activities '''
		# remove_activities = self.VARS.CONVERTION_STATIC
		# keep_activities = self.VARS.CONVERTION_DYNAMIC
		# self.data_set_DYNAMIC = input_data_window_large.read_data_sets_without_activity(subject_set, self.VARS.len_convertion_list(keep_activities), remove_activities, None, keep_activities)



	def initialize_networks(self):
		''' ORIGINAL GRAPH'''
		print 'Loading original network'
		config = self.VARS.get_config(900, 17, 10, 100, 'original')
		self.cnn_original = CNN.CNN_TWO_LAYERS(config)
		self.cnn_original.set_data_set(self.data_set_ORIGINAL)
		#self.cnn_original.load_model('models/original')

		print 'Loading Static dynamic network'
		''' STATIC DYNAMIC GRAPH'''
		config = self.VARS.get_config(900, 2, 10, 100, 'sd')
		self.cnn_sd = CNN.CNN_TWO_LAYERS(config)
		self.cnn_sd.load_model('models/sd')

		''' STATIC GRAPH'''
		print 'Loading static network'
		config = self.VARS.get_config(900, 5, 10, 100, 'static')
		self.cnn_static = CNN.CNN_TWO_LAYERS(config)
		self.cnn_static.load_model('models/static')

		''' DYNAMIC GRAPH'''
		print 'Loading dynamic network'
		config = self.VARS.get_config(900, 12, 10, 100, 'dynamic')
		self.cnn_dynamic = CNN.CNN_TWO_LAYERS(config)
		self.cnn_dynamic.load_model('models/dynamic')

	
	def run_network(self, index):
		''' ORIGINAL '''
		data = self.data_set_ORIGINAL.test.next_data_label(index)
		actual = np.argmax(data[1])+1
		#print 'Actual', actual

		''' STATIC DYNAMIC PREDICTION'''
		prediction = self.cnn_sd.run_network(data)
		#print "SD prediction", prediction
		if prediction == 2:
			''' STATIC PREDICTION '''
			prediction = self.cnn_static.run_network(data)
			prediction = self.VARS.RE_CONVERTION_STATIC.get(prediction)
			#print "STATIC prediction", prediction
		else:
			''' DYNAMIC PREDICTION '''
			prediction = self.cnn_dynamic.run_network(data)
			prediction = self.VARS.RE_CONVERTION_DYNAIC.get(prediction)
			#print "DYNAMIC prediction",prediction

		return actual == prediction


cnn_h = CNN_H()
cnn_h.initialize_networks()
score = 0.0
start = 0
end = 2000

for i in range(start,end):
	if cnn_h.run_network(i):
		score += 1

print score * 1.0 / (end-start)