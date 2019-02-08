from __future__ import print_function
import os 
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 2b: 2-Layer MLP for IRIS

@author - Alexander G. Ororbia II
'''

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return 0.0		
			
def computeGrad(X,y,theta,reg): # returns nabla
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	W3 = theta[4]
	b3 = theta[5]
	# WRITEME: write your code here to complete the routine
	dW = W * 0.0
	db = b * 0.0
	dW2 = W2 * 0.0
	db2 = b2 * 0.0
	dW3 = W3 * 0.0
	db3 = b3 * 0.0
	return (dW,db,dW2,db2,dW3,db3)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	scores = 0.0
	probs = 0.0
	
def create_mini_batch(X, y, start, end):
	# WRITEME: write your code here to complete the routine
	mb_x = None
	mb_y = None
	return (mb_x, mb_y)
		
def shuffle(X,y):
	ii = np.arange(X.shape[0])
	ii = np.random.shuffle(ii)
	X_rand = X[ii]
	y_rand = y[ii]
	X_rand = X_rand.reshape(X_rand.shape[1:])
	y_rand = y_rand.reshape(y_rand.shape[1:])
	return (X_rand,y_rand)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/iris_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# load in validation-set
path = os.getcwd() + '/data/iris_test.dat'
data = pd.read_csv(path, header=None) 
cols = data.shape[1]  
X_v = data.iloc[:,0:cols-1]  
y_v = data.iloc[:,cols-1:cols] 

X_v = np.array(X_v.values)  
y_v = np.array(y_v.values)
y_v = y_v.flatten()


# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters randomly
h = 100 # size of hidden layer
h2 = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,h2)
b2 = np.zeros((1,h2))
W3 = 0.01 * np.random.randn(h2,K)
b3 = np.zeros((1,K))
theta = (W,b,W2,b2,W3,b3)

# some hyperparameters
n_e = 100
n_b = 10
step_size = 0.01 #1e-0
reg = 0 #1e-3 # regularization strength

train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(n_e):
	X, y = shuffle(X,y) # re-shuffle the data at epoch start to avoid correlations across mini-batches
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	#          you can use the "check" variable to decide when to calculate losses and record/print to screen (as in previous sub-problems)

	# WRITEME: write the inner training loop here (1 full pass, but via mini-batches instead of using the full batch to estimate the gradient)
	s = 0
	while (s < num_examples):
		# build mini-batch of samples
		X_mb, y_mb = create_mini_batch(X,y,s,s + n_b)
		
		# WRITEME: gradient calculations and update rules go here
		
		s += n_b

print(' > Training loop completed!')
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {0}'.format(np.mean(predicted_class == y)))

scores, probs = predict(X_v,theta)
predicted_class = np.argmax(scores, axis=1)
print('validation accuracy: {0}'.format(np.mean(predicted_class == y_v)))

# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)