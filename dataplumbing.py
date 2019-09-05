
import numpy as np

class Dataset:
	def __init__(self, xs, ls, ys):
		self.xs = xs	# Store the features
		self.ls = ls	# Store the length of each sequence
		self.ys = ys	# Store the labels
		self.num_samples = len(ys)
		self.num_features = len(xs[0,0,:])
		self.max_length = len(xs[0,:,0])
		self.num_classes = 1
	def batch(self, batch_size):
		js = np.random.randint(0, self.num_samples, batch_size)
		return self.xs[js,:,:], self.ls[js], self.ys[js]

# Load data
#
import sys
sys.path.append(r'D:\CSE\Major Project\rwa-master\length_problem_1000\rwa_model')
import input_data

train = Dataset(input_data.xs_train, input_data.ls_train, input_data.ys_train)
test = Dataset(input_data.xs_test, input_data.ls_test, input_data.ys_test)

