import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1c: MLPs \& the XOR Problem

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

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
# WRITEME: write your code here to complete the routine
    eps = 1e-5
    nabla_n = []
# NOTE: you do not have to use any of the code here in your implementation...
    for i in range(len(theta)):
        param = theta[i]
        param_grad = np.zeros(param.shape)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
        # Initializing the parameters at (x+eps) and (x-eps)
            theta_plus_eps = theta
            theta_minus_eps = theta
            ix = it.multi_index
        # Evaluate function at x+eps i.e f(x+eps)
            theta_plus_eps[i][ix] = param[ix] + eps
            f_x_plus_eps = computeCost(X,y,theta_plus_eps,reg)
        # Reset theta
            theta[i][ix] = param[ix] - eps        
        # Evaluate function at x i.e f(x-eps)
            theta_minus_eps[i][ix] = param[ix] - eps
            f_x_minus_eps = computeCost(X,y,theta_minus_eps,reg)
        # Reset theta
            theta[i][ix] = param[ix] + eps
        # Finally gradient at x
            param_grad[ix] = (f_x_plus_eps - f_x_minus_eps)/(2*eps)
        # Iterating over all dimensions
            it.iternext()
        nabla_n.append(param_grad)
    return tuple(nabla_n)

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

np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/xor.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check! 
h = 6 #100 # size of hidden layer
W1 = 0.05 * np.random.randn(D,h) #0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h)) + 1.0
W2 = 0.05 * np.random.randn(h,K) #0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K)) + 1.0
theta = (W1,b1,W2,b2) 

# some hyperparameters
reg = 1e-3 # regularization strength

nabla_n = computeNumGrad(X,y,theta,reg)
nabla = computeGrad(X,y,theta,reg)
nabla_n = list(nabla_n)
nabla = list(nabla)

for jj in range(0,len(nabla)):
    is_incorrect = 0 # set to false
    grad = nabla[jj]
    grad_n = nabla_n[jj]
    err = np.linalg.norm(grad_n - grad) / (np.linalg.norm(grad_n + grad))
    if(err > 1e-7):
        print("Param {0} is WRONG, error = {1}".format(jj, err))
    else:
        print("Param {0} is CORRECT, error = {1}".format(jj, err))

# re-init parameters
h = 6 #100 # size of hidden layer
W1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = (W1,b1,W2,b2) 

# some hyperparameters
no_epochs = 100
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e0
reg = 0.0 # regularization strength

# gradient descent loop
for i in xrange(no_epochs):
# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    theta = (W1, b1, W2, b2)
    loss = computeCost(X,y,theta,reg)
    if i % check == 0:
        print "iteration %d: loss %f" % (i, loss)

    # perform a parameter update
    # WRITEME: write your update rule(s) here
    dW1, db1, dW2, db2 = computeGrad(X,y,theta,reg)
    W1 = W1 - step_size * dW1
    b1 = b1 - step_size * db1
    W2 = W2 - step_size * dW2
    b2 = b2 - step_size * db2
    
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f%%' % (100*np.mean(predicted_class == y))