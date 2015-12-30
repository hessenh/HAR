import tensorflow as tf

class STRUCT(object):
	"""docstring for CNN_Struct"""
	def __init__(self, 
					input_size, 
					output_size, 
					input_size_sqrt,
					bias_conv1,
					weight_conv1,
					bias_conv2,
					weight_conv2,
					bias_neural1,
					weight_neural1,
					downsampled_twice,
      				weight_neural_input1,
      				weight_neural_input2):

		print 'init network struc'
		'''Placeholders for input and output'''
		self.x = tf.placeholder("float", shape=[None, input_size])
		self.y_ = tf.placeholder("float", shape=[None, output_size])

		'''First convolutional layer'''
		self.W_conv1 = self.weight_variable([5, 5, 1, 32], 'W_conv1')
		self.b_conv1 = self.bias_variable([32], 'b_conv1')
		self.x_image = tf.reshape(self.x, [-1, input_size_sqrt, input_size_sqrt,1])
		self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)

		'''Second convolutional layer'''
		self.W_conv2 = self.weight_variable([5, 5, 32, 64], 'W_conv2')
		self.b_conv2 = self.bias_variable([64], 'b_conv2')
		self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = self.max_pool_2x2(self.h_conv2)

		'''Densly conected layer'''
		self.W_fc1 = self.weight_variable([weight_neural_input1, weight_neural1], 'W_fc1')
		self.b_fc1 = self.bias_variable([bias_neural1], 'b_fc1')
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, weight_neural_input1])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
		self.keep_prob = tf.placeholder("float")
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
		self.W_fc2 = self.weight_variable([weight_neural_input2, output_size], 'W_fc2')
		self.b_fc2 = self.bias_variable([output_size], 'b_fc2')

		self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

	def weight_variable(self,shape,name):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial,name= name)

	def bias_variable(self, shape, name):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name = name)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                    strides=[1, 2, 2, 1], padding='SAME')