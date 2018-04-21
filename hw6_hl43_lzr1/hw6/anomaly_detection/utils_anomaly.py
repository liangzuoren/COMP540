import numpy as np

def estimate_gaussian(X):
    
    """
    Estimate the mean and standard deviation of a numpy matrix X on a column by column basis
    """
    mu = np.zeros((X.shape[1],))
    var = np.zeros((X.shape[1],))
    ####################################################################
    #               YOUR CODE HERE                                     #
    ####################################################################
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=0)
    ####################################################################
    #               END YOUR CODE                                      #
    ####################################################################
    return mu, var


def select_threshold(yval,pval):
    """
    select_threshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    bestF1 = 0
    stepsize = (max(pval)-min(pval))/1000
    for epsilon in np.arange(min(pval)+stepsize, max(pval), stepsize):
        
        ####################################################################
        #                 YOUR CODE HERE                                   #
        ####################################################################
        # print type(yval), type(pval)
        yval = yval.flatten()
        pred = pval < epsilon
        # print yval.shape, pred.shape
        # print yval
        # print pred
        # print yval
        # print pred
        tp = sum((pred+yval)==2)
        fp = sum((pred-yval)==1)
        fn = sum((pred-yval)==-1)
        prec = np.true_divide(tp,(tp+fp))
        rec = np.true_divide(tp, (tp+fn))
        F1 = (2.0*prec*rec)/(prec+rec)
        # print ((pred+yval)==2).shape
        # print (pred+yval)==2
        # print sum((pred+yval)==2)
        # print tp
        # print fp
        # print fn
        # print prec, rec, 
        # print prec
        # print rec
        # print F1
        if F1 > bestF1:
            bestF1 = F1
            best_epsilon = epsilon

        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
