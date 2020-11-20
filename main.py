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
	c_vals = []
	for c in range(left,right,1):
		model = SVC(kernel=kernel, degree=2, max_iter=-1, C=(10**c))
		model.fit(x,y)
		y_truth = test_data[:,-1]
		y_pred = model.predict(test_data[:,:-1])
		accuracy = calc_accuracy(y_pred,y_truth)
		c_vals.append(c)
		print("\tAccuracy for C =", 10**c, " is :",accuracy)
		accuracies.append(accuracy)
	plt.plot(c_vals,accuracies)
	plt.xlabel("Value of log(C) base 10")
	plt.ylabel("Accuracy Percentage")
	print("\tMax Accuracy:",max(accuracies))
	print("")

def report_MLP(train_data, test_data, hidden_layers, range_l, range_r):
	x = train_data[:,:-1]
	y = train_data[:,-1]
	accuracies = []
	learning_rates = []    
	for c in range(range_l,range_r,1):
		model = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', learning_rate_init=10**c, max_iter=500)
		model.fit(x,y)
		y_truth = test_data[:,-1]
		y_pred = model.predict(test_data[:,:-1])
		accuracy = calc_accuracy(y_pred,y_truth)
		print("\tAccuracy for learning rate = ", 10**c, " is :",accuracy)
		learning_rates.append(c)
		accuracies.append(accuracy)
        
	plt.plot(learning_rates,accuracies)
	plt.xlabel("Value of log(learning rate) base 10")
	plt.ylabel("Accuracy Percentage")
	print("\tMax Accuracy:",max(accuracies))
	print("")
	return max(accuracies), 10**learning_rates[accuracies.index(max(accuracies))]

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
print("The input and output dimensions of MLP are: ",train_data.shape[1]-1," and 1")
print("")
print("With 0 hidden layers")
values = []
v0 = report_MLP(train_data, test_data, [],-5,0)
values.append(v0)
plt.show()

print("With 1 hidden layer of size 2")
v1 = report_MLP(train_data, test_data, [2],-5,0)
values.append(v1)
plt.show()

print("With 1 hidden layer of size 6")
v2 = report_MLP(train_data, test_data, [6],-5,0)
values.append(v2)
plt.show()

print("With 2 hidden layers of size (2,3)")
v3 = report_MLP(train_data, test_data, [2,3],-5,0)
values.append(v3)
plt.show()

print("With 2 hidden layers of size (3,2)")
v4 = report_MLP(train_data, test_data, [3,2],-5,0)
values.append(v4)
plt.show()

print(values)
max_v = 0.0
max_ind = 0.0
max_l = 0.0
for i in range(0,5):
	if(max_v<values[i][0]):
		max_v = values[i][0]
		max_ind = i
		max_l = values[i][1]
hidden_layers = []
hidden_layers.append("()")
hidden_layers.append("(2)")
hidden_layers.append("(6)")
hidden_layers.append("(2,3)")
hidden_layers.append("(3,2)")
print("Hyperparameters for the best model found in MLP: ")
print("Best Accuracy is: ",max_v)
print("Optimiser: Stochastic Gradient Descent Optimiser ")
print("Activation: relu")
print("Hidden Layers Dimensions: ", hidden_layers[max_ind])
print("Learning Rate = ", max_l)
print("Max iterations num: 500")
print("Batch Size: min(200,n_samples)(default)")
print("Tol: 0.0001(default)")
print("Epoch: 10(default)")
print("Other hyperparameters: default")
