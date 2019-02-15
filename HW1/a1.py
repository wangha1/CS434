# READ ME
## Program can be run directly using command "python a1.py"
### Please put all the input file under the same folder as this program
### note that matplotlib is failing to connect to any X server for its GTK display
### You can run a local X server and enable X11 forwarding in your ssh client, to display the output on your local machine. You can verify this is working by checking that the $DISPLAY environment variable is set on the server
### to avoid that error cause program fail to running, I comment all the plot part in this program, you can uncomment those to see the plot, but make sure that matplotlib works on your environment



import sys
import numpy as np
#import matplotlib.pyplot as plt
from numpy import linalg as LA
import math


def sigmoid(x):
	return 1/(1+np.exp(-x))


def part1():
	trainMatrix_x = []
	trainMatrix_y = []
	testMatrix_x = []
	testMatrix_y = []
	train = "housing_train.txt"
	test = "housing_test.txt"
	with open(train, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split()
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements.append(1)
			trainMatrix_x.append(elements)
			trainMatrix_y.append(y_element)
	f.close()
	with open(test, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split()
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements.append(1)
			testMatrix_x.append(elements)
			testMatrix_y.append(y_element)
	f.close()

	X = np.matrix(trainMatrix_x)
	Y = np.matrix(trainMatrix_y)

	W = X.T * X				#X.T is the transpose of X
	W = W.I					#W.I is the inverse of W
	W = W * X.T * Y
	print("-------------------------------------------------------------")
	print("Result that produce with dummy variable")
	print("Weights are: ")
	print(W)
	print("-------------------------------------------------------------")



	E = (Y - X * W).T * (Y - X * W)
	E_Ave = E / X.shape[0]	#X.shape[0] return the number of rows on X, in our case, the number of examples
	print("ASE of training data is: ")
	print(E_Ave)
	print("-------------------------------------------------------------")


	X1 = np.matrix(testMatrix_x)
	Y1 = np.matrix(testMatrix_y)
	E1 = (Y1 - X1 * W).T * (Y1 - X1 * W)
	E1_Ave = E1 / X1.shape[0]
	print("ASE of testing data is: ")
	print(E1_Ave)
	print("-------------------------------------------------------------")

	for row in trainMatrix_x:
		row.pop()				#remove the dummy variable

	for row in testMatrix_x:
		row.pop()

	X = np.matrix(trainMatrix_x)
	Y = np.matrix(trainMatrix_y)

	W = X.T * X
	W = W.I
	W = W * X.T * Y
	print("-------------------------------------------------------------")
	print("Result that produce without dummy variable")
	print("Weights are: ")
	print(W)
	print("-------------------------------------------------------------")



	E = (Y - X * W).T * (Y - X * W)
	E_Ave = E / X.shape[0]
	print("ASE of training data is: ")
	print(E_Ave)
	print("-------------------------------------------------------------")


	X1 = np.matrix(testMatrix_x)
	Y1 = np.matrix(testMatrix_y)
	E1 = (Y1 - X1 * W).T * (Y1 - X1 * W)
	E1_Ave = E1 / X1.shape[0]
	print("ASE of testing data is: ")
	print(E1_Ave)
	print("-------------------------------------------------------------")

	for row in trainMatrix_x:
		row.insert(1,1)			#add the dummy variable back, which we removed for the last part

	for row in testMatrix_x:
		row.insert(1,1)

	features = 1
	x_axex = []
	y_axex_training = []
	y_axex_testing = []
	while(features < 50):	# adding 1-50 random variables (add 1 each time)
		for row in trainMatrix_x:
			row.append(np.random.normal(30,5))
		for row in testMatrix_x:
			row.append(np.random.normal(30,5))

		X = np.matrix(trainMatrix_x)
		Y = np.matrix(trainMatrix_y)

		X1 = np.matrix(testMatrix_x)
		Y1 = np.matrix(testMatrix_y)
		
		W = X.T * X
		W = W.I
		W = W * X.T * Y
		
		E = (Y - X * W).T * (Y - X * W)
		E_Ave = E / X.shape[0]
		E1 = (Y1 - X1 * W).T * (Y1 - X1 * W)
		E1_Ave = E1 / X1.shape[0]
		
		x_axex.append(features)
		y_axex_training.append(E_Ave.item(0,0))
		y_axex_testing.append(E1_Ave.item(0,0))
		
		features += 1

#	plt.plot(x_axex,y_axex_training,label = "Training ASE")
#	plt.plot(x_axex, y_axex_testing, label = "testing ASE")
#	plt.legend()
#	print("Close graph to continue")
#	plt.show()




def part2_1():
	trainMatrix_x = []
	trainMatrix_y = []
	testMatrix_x = []
	testMatrix_y = []
	ite_x = []
	train_y = []
	test_y = []
	train = "usps-4-9-train.csv"
	test = "usps-4-9-test.csv"
	with open(train, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split(",")
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements = [float(i)/256 for i in elements]
			elements.append(1)
			trainMatrix_x.append(elements)
			trainMatrix_y.append(y_element)
	f.close()

	with open(test, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split(",")
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements = [float(i)/256 for i in elements]
			elements.append(1)
			testMatrix_x.append(elements)
			testMatrix_y.append(y_element)
	f.close()

	W = np.zeros((1,257))
	delta = np.zeros((1,257))
	delta_pre = np.zeros((1,257))
	ite = 1



	while(True):
		corr = 0

		for i in range(1400):
			xi = trainMatrix_x[i]
			yi = trainMatrix_y[i][0]
			y = sigmoid(np.dot(W,xi))
			y = np.array(y)
			err = yi-y
			delta += err * xi
			
			
			if(y >= 0.5 and yi == 1) or (y < 0.5 and yi == 0):
				corr+=1

		if(LA.norm(delta-delta_pre) <= 0.001 or corr >= 1400 or ite >= 1501):
			break;

		else:
			testcorr = 0
			for i in range(800):
				testxi = testMatrix_x[i]
				testyi = testMatrix_y[i][0]
				testy = sigmoid(np.dot(W,testxi))
				if(testy >= 0.5 and testyi == 1) or (testy < 0.5 and testyi == 0):
					testcorr+=1
			print("-------------------------------")
			print("iteration", ite)
			print("norm", LA.norm(delta-delta_pre))
			print("training data accuracy", float(corr/1400.0))
			print("testing data accuracy", float(testcorr/800.0))
			W += 0.00000005 * delta
			delta_pre = np.array(delta.tolist())
			ite_x.append(ite)
			ite+=1
			train_y.append(float(corr/1400.0))
			test_y.append(float(testcorr/800.0))

#	plt.plot(ite_x, train_y, label = "Training accuracy")
#	plt.plot(ite_x, test_y, label = "Testing accuracy")
#	plt.legend()
#	print("Close graph to continue")
#	plt.show()


def part2_2():
	trainMatrix_x = []
	trainMatrix_y = []
	testMatrix_x = []
	testMatrix_y = []
	ite_x = []
	train_y = []
	test_y = []
	train = "usps-4-9-train.csv"
	test = "usps-4-9-test.csv"
	with open(train, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split(",")
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements = [float(i)/256 for i in elements]
			elements.append(1)
			trainMatrix_x.append(elements)
			trainMatrix_y.append(y_element)
	f.close()

	with open(test, 'r') as f:
		temp = f.read().splitlines()
		for line in temp:
			elements = line.split(",")
			elements = [float(i) for i in elements]
			y_element = [elements.pop()]
			elements = [float(i)/256 for i in elements]
			elements.append(1)
			testMatrix_x.append(elements)
			testMatrix_y.append(y_element)
	f.close()
	
	W = np.zeros((1,257))
	delta = np.zeros((1,257))
	delta_pre = np.zeros((1,257))
	ite = 1
	
	
	
	while(True):
		corr = 0
		
		for i in range(1400):
			xi = trainMatrix_x[i]
			yi = trainMatrix_y[i][0]
			y = sigmoid(np.dot(W,xi))
			y = np.array(y)
			err = yi-y
			delta += err * xi
			
			
			if(y >= 0.5 and yi == 1) or (y < 0.5 and yi == 0):
				corr+=1
	
		if(LA.norm(delta-delta_pre) <= 0.001 or corr >= 1400 or ite >= 1501):
			break;

		else:
			testcorr = 0
			for i in range(800):
				testxi = testMatrix_x[i]
				testyi = testMatrix_y[i][0]
				testy = sigmoid(np.dot(W,testxi))
				if(testy >= 0.5 and testyi == 1) or (testy < 0.5 and testyi == 0):
					testcorr+=1
			print("-------------------------------")
			print("iteration", ite)
			print("norm", LA.norm(delta-delta_pre))
			print("training data accuracy", float(corr/1400.0))
			print("testing data accuracy", float(testcorr/800.0))
			W += 0.00000005 * (delta + 0.001 * W)
			delta_pre = np.array(delta.tolist())
			ite_x.append(ite)
			ite+=1
			train_y.append(float(corr/1400.0))
			test_y.append(float(testcorr/800.0))

#	plt.plot(ite_x, train_y, label = "Training accuracy")
#	plt.plot(ite_x, test_y, label = "Testing accuracy")
#	plt.legend()
#	print("Close graph to terminate")
#	plt.show()


part1()
part2_1()
part2_2()


895.5
