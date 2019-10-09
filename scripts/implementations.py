# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    
    return 1/2*np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss."""
    
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    e = y - tx.dot(w)
    g = -tx.T.dot(e) / len(e)
    return g, e


def least_squares(y, tx):
    """ Least squares regression using normal equations. """
    
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y,tx, w)
    return (w, loss)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent. """
    
    w = [initial_w]
    loss = []
    w_ = initial_w
    for n_iter in range(max_iters):
        g, e = compute_gradient(y, tx, w_)
        loss_ = calculate_mse(e)
        w_ = w_ - gamma * g
        w.append(w_)
        loss.append(loss_)
    return (w, loss)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset. """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent. """
    
    w = [initial_w]
    loss = []
    w_ = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            g, e = compute_gradient(y_batch, tx_batch, w_)
            w_ = w_ - gamma * g
            loss_ = compute_loss(y, tx, w_)
            w.append(w_)
            loss.append(loss_)

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations. """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w= np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_loss(y,tx,w)
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD. """
    raise NotImplementedError
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD. """
    raise NotImplementedError
    return (w, loss)


def transform_factor(x, column_idx, nlevels):
    """ Transform a column with categorical variables into different columns with binary data.
    E.g feature = [1, 2, 1, 3] -> features = [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]  """
   
    new_var = np.zeros((x.shape[0], nlevels))
    for i in range(x.shape[0]):
        for j in range(nlevels):
            if x[i,column_idx] == j:
                new_var[j,0] = 1
               
    x = np.delete(x, column_idx, axis = 1)
   
    return np.concatenate((x, new_var), axis=1)
    