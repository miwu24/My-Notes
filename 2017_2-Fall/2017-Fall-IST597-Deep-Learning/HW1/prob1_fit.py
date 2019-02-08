import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression

@author - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.023 # step size coefficient
eps = 0.001 # controls convergence criterion
n_epoch = 500 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
    return np.array([(theta[0]+theta[1]*X[i,0])[0] for i in list(range(0,X.shape[0]))])

def gaussian_log_likelihood(mu, y):
    # WRITEME: write your code here to complete the sub-routine
    return np.log(1/np.sqrt(2*np.pi)*np.e**(-(1/2)*(y-mu)**2))

def computeCost(X, y, theta):
    # loss is now Bernoulli cross-entropy/log likelihood
    b = theta[0]
    w = theta[1]
    m = X.shape[0]
    return (sum([(w[0]*X[i,0]+b[0]-y[i,0])**2/(2*m) for i in list(range(0,X.shape[0]))]))[0]

def computeGrad(X, y, theta): 
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking
    b = theta[0]
    w = theta[1]
    m = X.shape[0]
    dL_db = sum([(w[0]*X[i,0]+b[0]-y[i,0])*(1/m) for i in list(range(0,X.shape[0]))]) # derivative w.r.t. model weights w
    dL_dw = sum([(w[0]*X[i,0]+b[0]-y[i,0])*X[i,0]*(1/m) for i in list(range(0,X.shape[0]))]) # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# display some information about the dataset itself here
# WRITEME: write your code here to print out information/statistics about the data-set "data" using Pandas (consult the Pandas documentation to learn how)
# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
Losses = np.array([])
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
while(i < n_epoch):
    dL_db, dL_dw = computeGrad(X, y, theta)
    b = theta[0]
    w = theta[1]
    
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    b = b - alpha*dL_db
    w = w - alpha*dL_dw
    # (note: don't forget to override the theta variable...)
    theta = (b, w)
    
    L = computeCost(X, y, theta) # track our loss after performing a single step
    Losses = np.hstack((Losses,np.array([L])))
    print(" {0} L = {1}".format(i,L))
    i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

plt.savefig('Linear_Regression.png')
plt.show() # convenience command to force plots to pop up on desktop

# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
plt.plot(np.array(range(0,Losses.shape[0])),Losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('Epoch_Loss.png')
plt.show()