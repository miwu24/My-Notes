import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification

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
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 100 # regularization coefficient
alpha = 0.5 # step size coefficient
n_epoch = 500 # number of epochs (full passes through the dataset)
eps = 1.0e-9  # controls convergence criterion

# begin simulation

def sigmoid(z):
	# WRITEME: write your code here to complete the routine
    return 1/(1+(np.e)**(-z))

def predict(X_feat, theta):  
	# WRITEME: write your code here to complete the routine
    Reg = regress(X_feat, theta)
    return (Reg>1/2).astype(int)

def regress(X_feat, theta):
    # WRITEME: write your code here to complete the routine
    Theta = np.concatenate((theta[0],theta[1].T),axis = 0)
    Linear = np.dot(X_feat, Theta)
    return sigmoid(Linear)

def bernoulli_log_likelihood(p, y):
	# WRITEME: write your code here to complete the routine
    
	return -1.0
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    # WRITEME: write your code here to complete the routine
    f_theta_X = regress(X,theta)
    m = X.shape[0]
    return (1/(2*m))*(-y*np.log(f_theta_X)-(1-y)*np.log(1-f_theta_X)).sum()+beta/(2*m)*(theta[1]**2).sum()

def computeGrad(X, y, theta, beta): 
    # WRITEME: write your code here to complete the routine (
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    m = (X.shape)[0]
    Reg = regress(X,theta)
    dL_db = (1/(2*m))*(np.sum(Reg-y)) 
    dL_db = dL_db.reshape(1,1) # derivative w.r.t model bias b
    n = (theta[1].shape)[1]
    dL_dw = (1/(2*m))*np.array([np.sum((Reg-y)*X[:,j+1].reshape(m,1))+beta*theta[1][0,j] for j in range(0,n)]) 
    dL_dw = dL_dw.reshape(1,n) # derivative w.r.t. model weights w
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]
 
x1 = data2['Test 1']  
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
m = X2.shape[0]
X2 = np.concatenate((np.ones((m,1)),X2),axis = 1)
y2 = np.array(y2.values)
b = np.array([0]).reshape(1,1)
w = np.zeros((1,X2.shape[1]-1))
theta = (b, w)

L = computeCost(X2, y2, theta, beta)
print("-1 L = {0}".format(L))
i = 0
while(i < n_epoch):
    dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
    #print(theta)
    b = theta[0]
    w = theta[1]
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    b = b - alpha*dL_db
    w = w - alpha*dL_dw
    theta = (b, w)
    
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    if (abs(L-computeCost(X2, y2, theta, beta))<eps):
        break
    
    L = computeCost(X2, y2, theta, beta)
        
    print(" {0} L = {1}".format(i,L))
    i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

predictions = predict(X2, theta)
# compute error (100 - accuracy)
err = 0.0
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
err = (predictions!=y2).sum()/(y2.shape)[0]
print('Error = {0}'.format(err * 100.))


# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
for i in range(1, degree+1):  
	for j in range(0, i+1):
		feat = np.power(xx1, i-j) * np.power(yy1, j)
		if (len(grid_nl) > 0):
			grid_nl = np.c_[grid_nl, feat]
		else:
			grid_nl = feat
grid_nl = np.concatenate((np.ones((grid_nl.shape[0],1)),grid_nl), axis = 1)
probs = regress(grid_nl, theta).reshape(xx.shape)


f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
plt.title('beta = ' + str(beta) + ', error = ' + (str(err)[0:5]))
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.savefig('Logi_Reg_d_'+str(degree)+'_beta_'+str(beta)+'.png', dpi = 400)

plt.show()