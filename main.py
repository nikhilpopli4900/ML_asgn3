import csv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import math
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def pre_process() :
	data = np.loadtxt("spambase.DATA",delimiter=",")
	#print(data)
	#print(data.shape)
	return data

def calc_accuracy(y,ya):
	err = (y-ya)**2
	return 100 - np.mean(err)*100

def report_SVM(train_data, test_data, kernel, left, right):
	x = train_data[:,:-1]
	y = train_data[:,-1]
	accuracies = []
	for c in range(left,right,1):
		model = SVC(kernel=kernel, degree=2, max_iter=-1, C=(10**c))
		model.fit(x,y)
		y_truth = test_data[:,-1]
		y_pred = model.predict(test_data[:,:-1])
		accuracy = calc_accuracy(y_pred,y_truth)
		print("\tAccuracy for C =", 10**c, " is :",accuracy)
		accuracies.append(accuracy)
	plt.plot(accuracies)
	print("\tMax Accuracy:",max(accuracies))
	print("")

def report_MLP(train_data, test_data, hidden_layers, range_l, range_r):
	x = train_data[:,:-1]
	y = train_data[:,-1]
	accuracies = []
	for c in range(range_l,range_r,1):
		model = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', learning_rate_init=10**c, max_iter=500)
		model.fit(x,y)
		y_truth = test_data[:,-1]
		y_pred = model.predict(test_data[:,:-1])
		accuracy = calc_accuracy(y_pred,y_truth)
		print("\tAccuracy for learning rate = ", 10**c, " is :",accuracy)
		accuracies.append(accuracy)
	plt.plot(accuracies)
	print("\tMax Accuracy:",max(accuracies))
	print("")


data = pre_process()
data[:,:-1] = scale(data[:,:-1])
np.random.shuffle(data)
splitval = int(data.shape[0]*0.8)
train_data = data[:splitval,:]
test_data = data[splitval:,:]

######################### PART 1 ##############################
# For linear kernel function
# Best seems 1
print("With linear kernel function")
report_SVM(train_data,test_data,'linear',-3,2)
plt.show()
# For Quadratic kernel function
# Best seems 10
print("With quadratic kernel function")
report_SVM(train_data,test_data,'poly',-3,4)
plt.show()
# For linear kernel function
# Best seems 10
print("With radial kernel function")
report_SVM(train_data,test_data,'rbf',-3,4)
plt.show()

############################## PART 2 ############################
print("")
print("")
print("Calculating for MLP")
print("")
print("With 0 hidden layers")
report_MLP(train_data, test_data, [],-5,0)
plt.show()
print("With 1 hidden layer of size 2")
report_MLP(train_data, test_data, [2],-5,0)
plt.show()
print("With 1 hidden layer of size 6")
report_MLP(train_data, test_data, [6],-5,0)
plt.show()
print("With 2 hidden layers of size (2,3)")
report_MLP(train_data, test_data, [2,3],-5,0)
plt.show()
print("With 2 hidden layers of size (3,2)")
report_MLP(train_data, test_data, [3,2],-5,0)
plt.show()