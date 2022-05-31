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

    '''
    Answer Code
    N = X.shape[0]
    for i in range(N):
        scores = X[i].dot(W) # scores.shape is N x C

        # shift values for 'scores' for numeric reasons (over-flow cautious)
        scores -= scores.max()

        probs = np.exp(scores)/np.sum(np.exp(scores))

        loss += -np.log(probs[y[i]])

        # since dL(i)/df(k) = p(k) - 1 (if k = y[i]), where f is a vector of scores for the given example
        # i is the training sample and k is the class
        dscores = probs.reshape(1,-1)

        dscores[:, y[i]] -= 1

        # since scores = X.dot(W), iget dW by multiplying X.T and dscores
        # W is D x C so dW should also match those dimensions
        # X.T x dscores = (D x 1) x (1 x C) = D x C
        dW += np.dot(X[i].T.reshape(X[i].shape[0], 1), dscores)
        print(dW.shape)
        break
    '''
    ################################## My Code ############################################
    N,D = X.shape
    C = W.shape[1]
    
    for i in range(N):
      scores = X[i].dot(W)         #score = 1*C matrix
      scores -= scores.max()         #exp 계산할때 에러 방지

      probs = np.exp(scores)/np.sum(np.exp(scores))         #probs = 1*C matrix
      loss += -np.log(probs[y[i]])         # loss = constant


      # since dL(i)/df(k) = p(k) - 1 (if k = y[i]), where f is a vector of scores for the given example
      # i is the training sample and k is the class

      dScore = probs.reshape(1,-1)         #dScore = 1*C 모양으로 꼭 만들어 줘야 하나봄
      dScore = dScore-1
      dW += np.dot(X[i].T.reshape(X[i].shape[0], 1), dScore)         #(D*1) *(1*C) = (D*C)만드니깐 & dW = dScore * X
      
    loss /= N
    dW /= N

    loss += reg * np.sum(W * W)
    dW += 2*reg*W 
    #############################################################################
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
    N, D = X.shape
    C = W.shape[1]

    scores = X.dot(W)         #scores = N*C matrix
    scores -= scores.max(axis = 1, keepdims = True)         #max in each row
    
    probs = np.exp(scores)/np.sum(np.exp(scores), axis = 1, keepdims = True)         #probs = N*C matrix

    loss = -np.log(probs[np.arange(N), y])
    loss = np.sum(loss)   
    loss /= N
    loss += reg * (np.sum(W * W))

    dScores = probs.reshape(N, -1)         #dScores = N*C matrix
    dScores = dScores -1
    dW = np.dot(X.T, dScores)         #dW shoule be D*C, so multipy X transpose(D*N) with dScore(N*C)
    dW /= N
    dW += 2*reg*W 

    #############################################################################
    return loss, dW
