import os

import numpy as np
import tensorflow as tf
import geturl as url
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model settings
#
num_features = 1
max_steps = 1000
num_cells = 250
num_classes = 1
activation = tf.nn.tanh
initialization_factor = 1.0

# Training parameters
#
num_iterations = 200
batch_size = 1
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

# Inputs
#
x = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])	# Features
l = tf.placeholder(tf.int32, [batch_size])	# Sequence length
y = tf.placeholder(tf.float32, [batch_size])	# Labels

# Trainable parameters
#
s = tf.Variable(tf.random_normal([num_cells], stddev=np.sqrt(initialization_factor)))	# Determines initial state

W_g = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells))
	)
)
b_g = tf.Variable(tf.zeros([num_cells]))
W_u = tf.Variable(
	tf.random_uniform(
		[num_features, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+num_cells))
	)
)
b_u = tf.Variable(tf.zeros([num_cells]))
W_a = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0*initialization_factor/(num_features+2.0*num_cells))
	)
)

W_o = tf.Variable(
	tf.random_uniform(
		[num_cells, num_classes],
		minval=-np.sqrt(6.0*initialization_factor/(num_cells+num_classes)),
		maxval=np.sqrt(6.0*initialization_factor/(num_cells+num_classes))
	)
)
b_o = tf.Variable(tf.zeros([num_classes]))

# Internal states
#
n = tf.zeros([batch_size, num_cells])
d = tf.zeros([batch_size, num_cells])
h = tf.zeros([batch_size, num_cells])
a_max = tf.fill([batch_size, num_cells], -1E38)	

# Define model
#
h += activation(tf.expand_dims(s, 0))

for i in range(max_steps):

	if i==l:
		break
	x_step = x[:,i,:]
	xh_join = tf.concat(axis=1, values=[x_step, h])	

	u = tf.matmul(x_step, W_u)+b_u
	g = tf.matmul(xh_join, W_g)+b_g
	a = tf.matmul(xh_join, W_a)     

	z = tf.multiply(u, tf.nn.tanh(g))

	a_newmax = tf.maximum(a_max, a)
	exp_diff = tf.exp(a_max-a_newmax)
	exp_scaled = tf.exp(a-a_newmax)

	n = tf.multiply(n, exp_diff)+tf.multiply(z, exp_scaled)	
	d = tf.multiply(d, exp_diff)+exp_scaled	
	h_new = activation(tf.div(n, d))
	a_max = a_newmax

	h = tf.where(tf.greater(l, i), h_new, h)	

ly = tf.matmul(h, W_o)+b_o
ly_flat = tf.reshape(ly, [batch_size])
py = tf.nn.sigmoid(ly_flat)

def get_prediction(str):
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, "bin/train.ckpt")
		xs, ls, ys = url.get_url(str)
		feed = {x: xs, l: ls, y: ys} 
		classification = sess.run(py, feed_dict=feed)
		return round(classification[0])