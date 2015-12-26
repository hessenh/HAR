# Convolutional Neural Network
# 
# ==============================================================================


import tensorflow as tf
import numpy as np
import input_data_window_large

class CNN_TWO_LAYERS(object):
  def __init__(self, subject_set, config):
    self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

    '''Config'''
    self._input_size = config['input_size']
    self._output_size = config['output_size']
    self._iteration_size = config['iteration_size']
    self._batch_size = config['batch_size']
    self._save_model_name = config['model_name']
    self._load_model = config['model_name_load']

    self._input_size_sqrt = int(self._input_size ** 0.5)

    '''Data set'''
    self._data_set = input_data_window_large.read_data_sets(subject_set, self._output_size)


    '''Default values'''
    # Conv1 
    self._bias_conv1 = 32
    self._weight_conv1 = 32

    # Conv2 
    self._bias_conv2 = 64
    self._weight_conv2 = 64

    # Neural network
    self._bias_neural1 = 1024
    self._weight_neural1 = 1024
    self._downsampled_twice = self._input_size_sqrt / 4 + 1 
    self._weight_neural_input1 = self._downsampled_twice * self._downsampled_twice * self._weight_conv2

    self._weight_neural_input2 = 1024

    '''Placeholders for input and output'''
    self.x = tf.placeholder("float", shape=[None, self._input_size])
    self.y_ = tf.placeholder("float", shape=[None, self._output_size])


    self.sess = None

  @property
  def info(self):
    return "Info: " + self._info

  def data_set(self):
    return self._data_set

  def initialize_network(self):
    '''Initialize variables'''
    self.sess = tf.InteractiveSession()
    self.sess.run(tf.initialize_all_variables())
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



    '''First convolutional layer'''
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])


    x_image = tf.reshape(self.x, [-1,self._input_size_sqrt,self._input_size_sqrt,1])


    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    '''Second convolutional layer'''
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    '''Densly conected layer'''
    W_fc1 = weight_variable([self._weight_neural_input1, self._weight_neural1])
    b_fc1 = bias_variable([self._bias_neural1])

    h_pool2_flat = tf.reshape(h_pool2, [-1, self._weight_neural_input1])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



    self.keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)



    W_fc2 = weight_variable([self._weight_neural_input2, self._output_size])
    b_fc2 = bias_variable([self._output_size])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    self.init_op = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    self.saver = tf.train.Saver()

  def run_network(self):
    if self._load_model:
      '''Using model'''
      
      print 'Using model:',self._load_model
      # Add ops to save and restore all the variables.
      saver = tf.train.Saver()
      # Restore variables from disk.
      saver.restore(self.sess, self._load_model)

      print("test accuracy %g"%self.accuracy.eval(feed_dict={
          self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))
   
    else:
      '''Creating model'''
      self.sess.run(self.init_op)
      for i in range(self._iteration_size):
        batch = self._data_set.train.next_batch(self._batch_size)
        if i%50 == 0:
          train_accuracy = self.accuracy.eval(feed_dict={
              self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))
        self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
      print("test accuracy %g"%self.accuracy.eval(feed_dict={
          self.x: self._data_set.test.data, self.y_: self._data_set.test.labels, self.keep_prob: 1.0}))


      save_path = self.saver.save(self.sess, self._save_model_name)
      print("Model saved in file: %s" % save_path)




# Subjects
train_subjects = ["P03","P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
test_subjects = ["P11"]
subject_set = [train_subjects, test_subjects]

# Config
config = {
   'input_size': 900, # Number of inputs 
   'output_size': 62, # Number of ouptuts
   'iteration_size': 1000, # Number of training iterations
   'batch_size': 100, # Number of samples in each training iteration (batch)
   'model_name': "test_model", # New model name
   'model_name_load': None # If presented, model will be loaded
}
cnn = CNN_TWO_LAYERS(subject_set ,config)
cnn.initialize_network()
cnn.run_network()