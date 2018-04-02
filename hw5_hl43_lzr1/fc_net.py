import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of d, a hidden dimension of h, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'theta1' and 'theta1_0' and second     #
    # layer weights and biases using the keys 'theta2' and 'theta2_0.          #
    ############################################################################
    # 4 lines of code expected
    self.params["theta1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params["theta1_0"] = np.zeros((1,hidden_dim))
    self.params["theta2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["theta2_0"] = np.zeros((1,num_classes))
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (m, d_1, ..., d_k)
    - y: Array of labels, of shape (m,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (m, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    # Hint: unpack the weight parameters from self.params
    # 3 lines of code expected
    theta1,theta1_0,theta2,theta2_0 = self.params["theta1"], self.params["theta1_0"], self.params["theta2"], self.params["theta2_0"]
    
    #First layer
    first_out, first_cache = affine_relu_forward(X, theta1, theta1_0)
    #Second layer
    second_out, second_cache = affine_forward(first_out,theta2,theta2_0)
    
    scores = second_out
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 4-8 lines of code expected
    
    sftmax_loss, sftmax_grad = softmax_loss(scores,y)
    loss = sftmax_loss
    loss = loss + 0.5*self.reg*np.sum(theta1**2)
    loss = loss + 0.5*self.reg*np.sum(theta2**2)
    
    dx_1, grads["theta2"], grads["theta2_0"] = affine_backward(sftmax_grad,second_cache)
    dx_2, grads["theta1"], grads["theta1_0"] = affine_relu_backward(dx_1, first_cache)
    
    grads["theta1"] += self.reg*theta1
    grads["theta2"] += self.reg*theta2
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout as an option. For a network with L layers,
  the architecture will be
  
  {affine - - relu - [dropout]} x (L - 1) - affine - softmax
  
  where  dropout is  optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 2 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in theta1 and theta1_0 for the second layer use theta2 and theta2_0, etc.#
    # Weights should beinitialized from a normal distribution with standard    #
    # deviation equal to weight_scale and biases should be initialized to zero.#
    #                                                                          #
    ############################################################################
    # about 4 lines of code
    for layer_number in range(1,self.num_layers):
        weight =  "theta"+str(layer_number)
        bias =  "theta"+str(layer_number)+"_0"
        if layer_number==1:
            self.params[weight] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
            self.params[bias] = np.zeros((1,hidden_dims[0]))
        elif layer_number==self.num_layers-1:
            self.params[weight] = weight_scale * np.random.randn(hidden_dims[layer_number-2], num_classes)
            self.params[bias] = np.zeros((1,num_classes))
        else:
            self.params[weight] = weight_scale * np.random.randn(hidden_dims[layer_number-2], hidden_dims[layer_number-1])
            self.params[bias] = np.zeros((1,hidden_dims[layer_number-1]))
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.

    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # Cast all parameters to the correct datatype

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for  dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    ############################################################################
    layer_out = {}
    layer_cache = {}
    dropout_cache = {}
    
    X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
    for layer_number in range(1,self.num_layers):
        weight_name = "theta"+str(layer_number)
        bias_name = "theta"+str(layer_number)+"_0"
        weight = self.params[weight_name]
        bias = self.params[bias_name]
        if layer_number==1:
            layer_out[weight_name], layer_cache[weight_name] = affine_relu_forward(X,weight,bias)
            previous_weight_name = weight_name
            if self.use_dropout:
                layer_out[weight_name], dropout_cache[weight_name] = dropout_forward(layer_out[weight_name], self.dropout_param)
        elif layer_number==self.num_layers-1:
            layer_out[weight_name], layer_cache[weight_name] = affine_forward(layer_out[previous_weight_name],weight,bias)
        else:
            layer_out[weight_name], layer_cache[weight_name] = affine_relu_forward(layer_out[previous_weight_name],weight,bias)
            previous_weight_name = weight_name
            if self.use_dropout:
                layer_out[weight_name], dropout_cache[weight_name] = dropout_forward(layer_out[weight_name], self.dropout_param)
                
    
    weight_name = "theta"+str(self.num_layers-1)
    
    scores = layer_out[weight_name]
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    sftmax_loss, sftmax_grad = softmax_loss(scores,y)
    loss = sftmax_loss
    for layer_number in range(1,self.num_layers):
        weight_name =  "theta"+str(layer_number)
        bias_name =  "theta"+str(layer_number)+"_0"
        weight = self.params[weight_name]
        bias = self.params[bias_name]
        loss = loss + 0.5*self.reg*np.sum(weight**2) #adding nth layer regularization 
    
    for layer_number in range(self.num_layers-1,0,-1):
        weight_name =  "theta"+str(layer_number)
        bias_name =  "theta"+str(layer_number)+"_0"
        weight = self.params[weight_name]
        bias = self.params[bias_name]
    
        if layer_number==self.num_layers-1:
            dx, grads[weight_name], grads[bias_name] = affine_backward(sftmax_grad,layer_cache[weight_name])
        else:
            if self.use_dropout:
                dx = dropout_backward(dx,dropout_cache[weight_name])
            dx, grads[weight_name], grads[bias_name] = affine_relu_backward(dx,layer_cache[weight_name])
        
        grads[weight_name] += self.reg*weight
            
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
