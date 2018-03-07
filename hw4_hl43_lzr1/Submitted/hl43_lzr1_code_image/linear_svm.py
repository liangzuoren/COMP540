import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  h = np.matmul(X,theta)
  loss = y*h;
  maxloss = C/float(m)*np.sum(np.maximum(np.zeros(m),1-loss))
  J = 0.5/float(m)*np.sum(theta**2)+maxloss
  comparison_vector = np.ones(m)
  gradloss = X*y[:,np.newaxis]
  grad = theta/float(m) + C/float(m)*np.sum(-gradloss[loss<comparison_vector,:],axis=0)
  grad = grad.T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  for i in range(m):
  		scores = X[i].dot(theta) # 1 x K vector of scores
  		correct_class_score = scores[y[i]]
  		for j in range(K):
  			if j == y[i]: # correct class
  				continue
  			margin = scores[j] - correct_class_score + delta
  			if margin > 0:
  				dtheta[:, y[i]] += -X[i] # gradient update for correct rows
  				dtheta[:, j] += X[i] # gradient update for incorrect rows
  				J += margin # loss update
  J /= m #average loss
  dtheta /= m #average gradient
  J += 0.5 * reg * np.sum(theta * theta) #add regularization term to loss
  dtheta += reg * theta #add regularization term to gradient


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################
  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  scores = X.dot(theta) # m x K vector of scores
  Y = np.zeros(scores.shape) # initialize the true class assignment matrix 
  # fill in the true class of each example (each row)
  for i,row in enumerate(Y):
  	row[y[i]] = 1
  Correct_class_scores = np.array( [ [ scores[i][y[i]] ]*K for i in range(m) ] )  
  Margin = scores - Correct_class_scores + ((scores - Correct_class_scores) != 0) * delta # margin = 0 if score = correct_class score
  X_with_margin_count = np.multiply(X.T , ( Margin > 0).sum(1) ).T

  J += np.sum((Margin>0)*Margin)/m
  J += 0.5 * reg * np.sum(theta * theta)
  dtheta += ( Margin > 0 ).T.dot(X).T/m
  dtheta -= (Margin == 0).T.dot(X_with_margin_count).T/m
  dtheta += reg*theta
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
