# Train different CNN
# 
# ==============================================================================

import input_data_window_large
import CNN
import CNN_STATIC_VARIABLES


class CNN_TRAIN(object):
   """docstring for CNN_H"""
   def __init__(self, network_type, iterations, window):
      self.VARS = CNN_STATIC_VARIABLES.CNN_STATIC_VARS()
      subject_set = self.VARS.get_subject_set()
      

      if network_type == 'sd':
         self.config = self.VARS.get_config(900, 2, iterations, 100, network_type)
         convertion = self.VARS.CONVERTION_STATIC_DYNAMIC
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      
      if network_type == 'original':
         self.config = self.VARS.get_config(900, 17, iterations, 100, network_type)
         convertion = self.VARS.CONVERTION_ORIGINAL
         print 'Creating data set'
         self.data_set = input_data_window_large.read_data_sets(subject_set, self.VARS.len_convertion_list(convertion), convertion, None, window)
      
      if network_type == 'static':
         self.config = self.VARS.get_config(900, 5, iterations, 100, network_type)
         remove_activities = self.VARS.REMOVE_DYNAMIC_ACTIVITIES
         keep_activities = self.VARS.CONVERTION_STATIC
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)

      if network_type == 'dynamic':
         remove_activities = self.VARS.CONVERTION_STATIC
         keep_activities = self.VARS.CONVERTION_DYNAMIC
         self.config = self.VARS.get_config(900, len(keep_activities), iterations, 100, network_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)
      
      if network_type == 'shuf_stand':
         remove_activities = self.VARS.CONVERTION_SHUF_STAND_INVERSE
         keep_activities = self.VARS.CONVERTION_SHUF_STAND
         self.config = self.VARS.get_config(900, len(keep_activities), iterations, 100, network_type)
         self.data_set = input_data_window_large.read_data_sets_without_activity(subject_set, len(keep_activities), remove_activities, None, keep_activities, window)
      

      self.cnn = CNN.CNN_TWO_LAYERS(self.config)
      self.cnn.set_data_set(self.data_set)
      self.cnn.train_network()
      self.cnn.save_model('models/' + network_type)


cnn_h = CNN_TRAIN('shuf_stand', 500, '1.5')