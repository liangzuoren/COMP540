from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X-mu)/sigma
   
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    for i in range(1,num_examples+1):
        Xtraining = X[0:i,:]
        ytraining = y[0:i]
        reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
        theta_opt0 = reglinear_reg1.train(Xtraining,ytraining,reg,num_iters=1000)
        error_train[i-1] = reglinear_reg1.loss(theta_opt0,Xtraining,ytraining,0)
        error_val[i-1] = reglinear_reg1.loss(theta_opt0,Xval,yval,0)

    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
  reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
  error_train = np.zeros((len(reg_vec),))
  error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
  for i in range(0,len(reg_vec)):
      reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
      theta_opt0 = reglinear_reg1.train(X,y,reg_vec[i],num_iters=1000)
      error_train[i] = reglinear_reg1.loss(theta_opt0,X,y,0)
      error_val[i] = reglinear_reg1.loss(theta_opt0,Xval,yval,0)
    
  return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    numTries = 50
    reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
    unaveraged_error_train = np.zeros((numTries,))
    unaveraged_error_val = np.zeros((numTries,))
    
    for i in range(0,num_examples):
        for j in range(0,numTries):
            shuffledSamples = np.random.permutation(num_examples)
            testSetNumbers = shuffledSamples[0:i+1]
    
            Xtraining = X[testSetNumbers,:]
            ytraining = y[testSetNumbers]
            theta_opt0 = reglinear_reg1.train(Xtraining,ytraining,reg,num_iters = 1000)
        
            Xvalidation = Xval[testSetNumbers,:]
            yvalidation = yval[testSetNumbers]
                
            unaveraged_error_train[j] = reglinear_reg1.loss(theta_opt0,Xtraining,ytraining,0)
            unaveraged_error_val[j] = reglinear_reg1.loss(theta_opt0,Xvalidation,yvalidation,0)
    
        error_train[i] = np.mean(unaveraged_error_train)
        error_val[i] = np.mean(unaveraged_error_val)

    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval








