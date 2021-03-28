from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    scores = X @ W # shape: (N, C), also called logits.
    scores -= np.max(scores) # to avoid numeric instability (http://cs231n.github.io/linear-classify/#softmax)

    exps = np.exp(scores) # shape: (N, C)
    sums = np.sum(exps, axis=1, keepdims=True) # shape: (N, 1)

    for i in range(N):
      loss += (-1) * exps[i, y[i]] + np.sum(exps[i])

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    # based on the chain rule and quotient rule
    y_true = np.zeros_like(scores)
    y_true[np.arange(N), y] = 1

    y_pred = exps / sums
    
    df = X.T # shape: (D, N), df(W)/dW = d(X @ W)/dW = X.T
    dL = y_pred - y_true # shape: (N, C), dL(f)/df
    # the gradient of Loss function with respect to weights
    dW = df @ dL 

    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    scores = X @ W # shape: (N, C), also called logits.
    scores -= np.max(scores) # to avoid numeric instability (http://cs231n.github.io/linear-classify/#softmax)

    exps = np.exp(scores) # shape: (N, C)
    sums = np.sum(exps, axis=1, keepdims=True) # shape: (N,)

    # since to finish without loop
    sum_of_exps = np.sum(scores[np.arange(N), y])
    sum_of_sums = np.sum(sums) 
    loss = (-1) * sum_of_exps + sum_of_sums

    # calculate the gradients
    y_true = np.zeros_like(scores)
    y_true[np.arange(N), y] = 1

    y_pred = exps / sums
    
    df = X.T # shape: (D, N), df(W)/dW = d(X @ W)/dW = X.T
    dL = y_pred - y_true # shape: (N, C), dL(f)/df
    dW = df @ dL 

    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
