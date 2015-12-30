import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data	

class Simple_network(object):
	"""docstring for Simple_network"""
	def __init__(self):
		super(Simple_network, self).__init__()

		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		self.x = tf.placeholder(tf.float32, [None, 784])

		self.W = tf.Variable(tf.zeros([784, 10]), name = "W")
		self.b = tf.Variable(tf.zeros([10]), name = "b")

		self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

		self.y_ = tf.placeholder(tf.float32, [None, 10])

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))


		self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

		self.init = tf.initialize_all_variables()

	def save_variables(self, model):
		#saver = tf.train.Saver({"W": self.W, "b": self.b})
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, model)
  		print("Model saved in file: %s" % save_path)

	def load_session(self, model):
		self.sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(self.sess, model)
		#self.sess.run(self.init)

	def train_network(self):
		self.sess = tf.Session()
		self.sess.run(self.init)
		for i in range(1000):
		  batch_xs, batch_ys = self.mnist.train.next_batch(100)
		  self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

	def test_network(self):
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(self.sess.run(accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}))

nn = Simple_network()
#nn.train_network()
#nn.save_variables("model.ckpt")

nn.load_session("model2.ckpt")
nn.test_network()

nn.load_session("model.ckpt")
nn.test_network()

nn.load_session("model2.ckpt")
nn.test_network()