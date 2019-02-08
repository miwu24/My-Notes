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
Problem 2a: 1-Layer MLP for IRIS

@author - Min-Chun Wu
'''
def softmax_loss(X, y):
# Forward pass
    N = X.shape[0]
    X -= np.max(X, axis=1, keepdims=True)
    exp_vals = np.exp(X)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[range(N), y]))
# Backward pass
    dX = np.array(probs, copy=True)
    dX[range(N), y] -= 1
    dX /= N
    return loss, probs, dX

def computeCost(X,y,theta,reg):
# WRITEME: write your code here to complete the routine
    W1, b1, W2, b2 = theta[0], theta[1], theta[2], theta[3]
    z = X.dot(W1) + b1       # FC1
    h = np.maximum(0, z)     # ReLU
    f = h.dot(W2) + b2       # FC2
    data_loss, _, _ = softmax_loss(f, y) # Softmax
    reg_loss = 0.5 * reg * np.sum(W1**2) + 0.5 * reg * np.sum(W2**2)
    loss= data_loss + reg_loss
    return loss

def computeGrad(X,y,theta,reg): # returns nabla
    W1, b1, W2, b2 = theta[0], theta[1], theta[2], theta[3]
    z = X.dot(W1) + b1         # FC1
    h = np.maximum(0, z)       # ReLU
    f = h.dot(W2) + b2         # FC2
    _, _, df = softmax_loss(f, y) # Softmax
    dh = df.dot(W2.T)
    dz = np.array(dh, copy=True)
    dz[z <= 0] = 0
# WRITEME: write your code here to complete the routine
    dW2 = np.dot(h.T, df) + reg * W2
    db2 = np.sum(df, axis=0)
    dW1 = np.dot(X.T, dz) + reg * W1
    db1 = np.sum(dz, axis=0)
    return (dW1,db1,dW2,db2)

def predict(X,theta):
# WRITEME: write your code here to complete the routine
    W1, b1, W2, b2 = theta[0], theta[1], theta[2], theta[3]
    z = X.dot(W1) + b1         # FC1
    h = np.maximum(0, z)       # ReLU
    scores = h.dot(W2) + b2         # FC2
    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return (scores,probs)

def create_mini_batch(X, y, start, end):
# WRITEME: write your code here to complete the routine
    mb_x = X[start : end, :]
    mb_y = y[start : end]
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
W1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = (W1,b1,W2,b2)

# some hyperparameters
n_e = 100
n_b = 10
check = 10
step_size = 1e-1 #1e-0
reg = 1e-3 #1e-3 # regularization strength

train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(n_e):
    X, y = shuffle(X,y) # re-shuffle the data at epoch start to avoid correlations across mini-batches
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    #          you can use the "check" variable to decide when to calculate losses and record/print to screen (as in previous sub-problems)
    train_loss = computeCost(X,y,theta,reg)
    valid_loss = computeCost(X_v,y_v,theta,reg)
    train_cost.append(train_loss)
    valid_cost.append(valid_loss)
    if i % check == 0:
        s = "iteration %d: training loss = %.2f, validation loss = %.2f" % (i, train_loss, valid_loss)
        print (s)
# WRITEME: write the inner training loop here (1 full pass, but via mini-batches instead of using the full batch to estimate the gradient)
    s = 0
    while (s < num_examples):
        # build mini-batch of samples
        X_mb, y_mb = create_mini_batch(X,y,s,s + n_b)
        # WRITEME: gradient calculations and update rules go here
        theta = (W1, b1, W2, b2)
        dW1, db1, dW2, db2 = computeGrad(X_mb,y_mb,theta,reg)
        W1 = W1 - step_size * dW1
        b1 = b1 - step_size * db1
        W2 = W2 - step_size * dW2
        b2 = b2 - step_size * db2
        
        s += n_b

print(' > Training loop completed!')
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {0}'.format(100*np.mean(predicted_class == y)))

scores, probs = predict(X_v,theta)
predicted_class = np.argmax(scores, axis=1)
print('validation accuracy: {0}'.format(100*np.mean(predicted_class == y_v)))

# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)
plt.plot(range(n_e), train_cost, range(n_e), valid_cost)
plt.legend(['training loss', 'validation loss'])
plt.show()