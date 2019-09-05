import numpy as np
import csv
import os

_path = r'D:\CSE\Major Project\rwa-master\length_problem_1000\rwa_model'
os.makedirs(_path+'/bin', exist_ok=True)

num_total = 57524
num_train = 50000
max_length = 1000
num_features = 1

xs_train = []
ls_train = []
ys_train = []
str = ""

with open("final_dataset.csv") as f:
	readCSV = csv.reader(f,delimiter=',')
	index = 1
	for row in readCSV:
		if index>num_train:
			continue
		string = row[0]
		label = row[1]
		ys_train.append(label)
		l = list(string)
		ls_train.append(len(l))
		for i in range(len(l),max_length):
			l.append('0')
		string = "".join(l)
		xs_temp = np.array(l)
		xs_temp = xs_temp.view(np.uint8)
		xs_temp = xs_temp.view(np.int)
		xs_train.append(xs_temp)
		index = index+1
		
xs_train = np.array(xs_train)

if not os.path.isfile(_path+'/bin/xs_train.npy'):
	xs_train = xs_train.reshape(num_train,max_length,num_features)
	np.save(_path+'/bin/xs_train.npy', xs_train)
else:
	xs_train = np.load(_path+'/bin/xs_train.npy')

if not os.path.isfile(_path+'/bin/ls_train.npy'):
	ls_train = np.array(ls_train,dtype=int)
	np.save(_path+'/bin/ls_train.npy', ls_train)
else:	
	ls_train = np.load(_path+'/bin/ls_train.npy')

ys_train = np.array(ys_train,dtype=int)

xs_test = []
ls_test = []
ys_test = []
str = ""

with open("final_dataset.csv") as f:
	readCSV = csv.reader(f,delimiter=',')
	index = 0
	for row in readCSV:
		index = index+1
		if index<=num_train:
			continue
		string = row[0]
		label = row[1]
		ys_test.append(label)
		l = list(string)
		ls_test.append(len(l))
		for i in range(len(l),max_length):
			l.append('0')
		string = "".join(l)
		xs_temp = np.array(l)
		xs_temp = xs_temp.view(np.uint8)
		xs_temp = xs_temp.view(np.int)
		xs_test.append(xs_temp)
		
xs_test = np.array(xs_test)

if not os.path.isfile(_path+'/bin/xs_test.npy'):
	xs_test = xs_test.reshape(num_total-num_train,max_length,num_features)
	np.save(_path+'/bin/xs_test.npy', xs_test)
else:
	xs_test = np.load(_path+'/bin/xs_test.npy')
	
if not os.path.isfile(_path+'/bin/ls_test.npy'):
	ls_test = np.array(ls_test,dtype=int)
	np.save(_path+'/bin/ls_test.npy', ls_test)
else:
	ls_test = np.load(_path+'/bin/ls_test.npy')

ys_test = np.array(ys_test,dtype=int)

