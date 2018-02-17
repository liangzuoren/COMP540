import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  
  for i in range(0,m):
      collected_bottomTerm = 0.0
      for j in range(0,theta.shape[1]):
          thetaX = np.dot(theta[:,j],X[i,:])
          bottomTerm = np.exp(thetaX)
          collected_bottomTerm = collected_bottomTerm + bottomTerm
      for k in range(0,theta.shape[1]):
          if(y[i] == k):
                I = 1.0
          else:
                I = 0.0
          thetaX = np.dot(theta[:,k],X[i,:])
          J = J + I*np.log(np.exp(thetaX)/collected_bottomTerm);
  J = -1.0/m * J 
  for j in range(0,dim):
      J = J + reg/(2.0*m)*np.sum(theta[j,:]**2)
  
  for k in range(0,theta.shape[1]):
      for i in range(0,m):
          collected_bottomTerm = 0.0
          for j in range(0,theta.shape[1]):
              thetaX = np.dot(theta[:,j],X[i,:])
              bottomTerm = np.exp(thetaX)
              collected_bottomTerm = collected_bottomTerm + bottomTerm
          if(y[i] == k):
                I = 1.0
          else:
                I = 0.0
          thetaX = np.dot(theta[:,k],X[i,:])
          grad[:,k] = grad[:,k] + np.dot(X[i,:],(I - np.exp(thetaX)/collected_bottomTerm))
      grad[:,k] = -1.0/m * grad[:,k] + reg/m*theta[:,k]   
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  #Matrix multiplication version of the code
  #Building Identity Matrix
  #theta_index = np.arange(0,theta.shape[1],dtype=np.float)[np.newaxis]
  #K_index = np.repeat(theta_index,m,axis=0)
  #Matching y vector across the thetas to get I 
  #y_vector = y[np.newaxis]
  #y_matrix = np.transpose(np.repeat(y_vector,theta.shape[1],axis=0))  
  #I = (K_index==y_matrix).astype(float)
  #Perform matrix multiplication for X^T*theta
  #thetaX = np.matmul(X,theta)
  #np.max(thetaX)
  #Subtracting the maximum to maintain numerical stability on the exp function
  #thetaX = thetaX - np.max(thetaX, axis=1).reshape(-1, 1)
  #Calculating cost function from the formula
  #J = -1.0/m * np.sum(np.sum(I*np.log(np.exp(thetaX)/np.transpose(np.sum(np.exp(thetaX),axis=1) [np.newaxis])),axis=0))
  #J = J + reg/(2.0*m)*np.sum(np.sum(theta**2,axis=0))
  
  #Calculating gradient
  #thetaX = np.transpose(np.matmul(X,theta))
  #Subtracting maximum to attempt to maintain numerical stability
  #thetaX = thetaX - np.max(thetaX, axis=1).reshape(-1, 1)
  #grad = np.transpose(np.matmul(np.transpose(I)-(np.exp(thetaX)/np.sum(np.exp(thetaX),axis=0)),X))
  #grad = -1.0/m * grad + reg/m*theta

  #Cleaner vector version of the code
  #Theta is d x K, X is m x d, y is m x 1
  thetaX = np.dot(X,theta)
  thetaX = thetaX - np.max(thetaX,axis=0)

  #ThetaX is m x K
  class_scores = np.exp(thetaX)
  
  #probability is m x K
  #keepdims to subtract from each row
  probability = class_scores/np.sum(class_scores,axis=1,keepdims = True)
  
  #Filtering by indices where y match the class
  probs_afterI =  probability[range(m),y]
  log_probs = np.log(probs_afterI)
  
  #calculate cost function
  J = -np.sum(log_probs)/m

  #adding regularization
  J = J + reg/(2.0*m)*np.sum(theta**2)
  
  #calculating gradient
  grad_scores = probability
  #Subtracting the I matrix from the gradient
  grad_scores[range(m),y] = grad_scores[range(m),y]-1
  grad_scores = grad_scores/m

  #multiplying X with rest of gradient matrix
  grad = np.dot(X.T, grad_scores)
  
  #Add regularization
  grad = grad + reg/m*theta  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
