import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1b: Softmax Regression \& the Spiral Problem

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

def computeGrad(X,y,theta,reg): # returns nabla
# WRITEME: write your code here to complete the routine
    W, b = theta[0], theta[1]
    f = X.dot(W) + b
    _, _, df = softmax_loss(f,y)
    dW = np.dot(X.T, df) + reg * W
    db = np.sum(df, axis=0)
    return (dW,db)

def computeCost(X,y,theta,reg):
# WRITEME: write your code here to complete the routine
    W, b = theta[0], theta[1]
    N = X.shape[0]
    f = X.dot(W) + b
    data_loss, _, _ = softmax_loss(f,y)
    reg_loss = 0.5 * reg * np.sum(W**2)
    cost = data_loss + reg_loss
    return cost

def predict(X,theta):
# WRITEME: write your code here to complete the routine
    W, b = theta[0], theta[1]
# evaluate class scores
    scores = X.dot(W) + b
    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return scores, probs

np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
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

#Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
theta = (W,b)

# some hyperparameters
no_epochs = 100
check = 10 # every so many pass/epochs, print loss/error to terminal
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 9.5
reg = 0 # regularization strength

# gradient descent loop
for i in xrange(no_epochs):
# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    theta = (W, b)
    loss = computeCost(X,y,theta,reg)
    if i % check == 0:
        print "iteration %d: loss %f" % (i, loss)

    # perform a parameter update
    # WRITEME: write your update rule(s) here
    dW, db = computeGrad(X,y,theta,reg)
    W = W - step_size*dW
    b = b - step_size*db
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 

# evaluate training set accuracy
scores, probs = predict(X,theta)
#scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f%%' % (100*np.mean(predicted_class == y))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_linear.png')

plt.show()