import numpy as np

class url:
	def __init__(self,xs,ls,ys):
		self.xs = xs
		self.ls = ls
		self.ys = ys
	def inputurl(self, batch_size):
		return self.xs, self.ls, self.ys
	

def get_url(string):
	x = []
	l = []
	y = []
	str = string
	lis = list(str)
	l.append(len(str))
	l = np.array(l,dtype=int)
	y.append(1)
	y = np.array(y,dtype=int)
	for i in range(len(lis),1000):
		lis.append('0')
	xs_temp = np.array(lis)
	xs_temp = xs_temp.view(np.uint8)
	xs_temp = xs_temp.view(np.int)
	x.append(xs_temp)
	x = np.array(x,dtype=int)
	x = x.reshape(1,1000,1)
	obj = url(x,l,y)
	return obj.inputurl(1)