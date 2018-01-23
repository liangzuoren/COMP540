import numpy as np
import scipy

class RegularizedLinearRegressor_Multi:

    def __init__(self):
        self.theta = None


    def train(self,X,y,reg=1e-5,num_iters=100):

        """
        Train a linear model using regularized  gradient descent.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing


        Outputs:
        optimal value for theta
        """
    
        num_train,dim = X.shape
        theta = np.ones((dim,))


        # Run scipy's fmin algorithm to run the gradient descent
        theta_opt = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(X,y,reg),maxiter=num_iters)
            
        
        return theta_opt

    def loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        
        pass

    def grad_loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        pass

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is a real number.
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted outputs in y_pred.           #
        #  1 line of code expected                                                #
        ###########################################################################
        y_pred = np.dot(X,self.theta)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def normal_equation(self,X,y,reg):
        """
        Solve for self.theta using the normal equations.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Solve for theta_n using the normal equation.                            #
        #  One line of code expected                                              #
        ###########################################################################

        theta_n = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
        self.theta = theta_n

        ###########################################################################
        return theta_n

class RegularizedLinearReg_SquaredLoss(RegularizedLinearRegressor_Multi):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,*args):
        theta,X,y,reg = args

        num_examples,dim = X.shape
        J = 0
        grad = np.zeros((dim,))
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) wrt to X,y, and theta.                               #
        #  2 lines of code expected                                               #
        ###########################################################################
        #J = 1/(2*float(num_examples))*sum(np.square(np.dot(X,theta)-y)) + reg/(2*float(num_examples))*(np.dot(theta,theta))
    
        hypothesis = y-np.sum((X*theta),axis=1)
        J = 1.0/(2*num_examples)*sum((hypothesis**2)) + reg/(2.0*num_examples)*sum(theta**2)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self,*args):                                                                          
        theta,X,y,reg = args
        num_examples,dim = X.shape
        grad = np.zeros((dim,))

        ###########################################################################
        # TODO:                                                                   #
        # Calculate gradient of loss function wrt to X,y, and theta.              #
        #  3 lines of code expected                                               #
        ###########################################################################
        #grad0 = 1/float(num_examples)*np.dot(np.dot(theta,X.T)-y,X[:,0])
        #gradj = 1/float(num_examples)*np.dot(np.dot(theta,X.T)-y,X[:,1:dim]) + reg/float(num_examples)*theta[1:dim]
        #grad = np.append(grad0, gradj)
        #[1:len(gradj)]
        grad0 =  1.0/num_examples*sum((np.sum((X*theta),axis=1)-y)*X[:,0])
        gradj = 1.0/num_examples*np.sum((np.sum((X*theta),axis=1)-y)*X[:,1:dim].T,axis=1) + reg/num_examples*theta[1:dim]
        grad = np.append(grad0,gradj)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad
