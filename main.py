import input_data
import tensorflow as tf

data = input_data.read_data_sets("test/", one_hot=True)

print data.__dict__
print len(data.train._images[0])

# Input variables
x = tf.placeholder(tf.float32, [None, 94])

# Weigths and bias
W = tf.Variable(tf.zeros([94, 17]))
b = tf.Variable(tf.zeros([17]))

# Softmax function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Labels
y_ = tf.placeholder(tf.float32, [None, 17])


# Cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# Training
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# Initialize 
init = tf.initialize_all_variables()

# Session
sess = tf.Session()
sess.run(init)

for i in range(1):
    batch_xs, batch_ys = data.train.next_batch(10)
    
    print len(batch_ys),len(batch_xs)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

