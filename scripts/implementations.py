# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np

# %% Standarization and Mini-batch

def standardize(x):
    """ Standardize the original data set."""
    
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    
    return x


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

# %% Loss function and Gradient

def compute_mse(y, tx, w):
    """ Calculate the mse for vector e."""
    
    e = y - tx.dot(w)
    
    return e.T.dot(e) / (2*len(y))


def compute_gradient(y, tx, w):
    """ Compute the gradient."""
    
    e = y - tx.dot(w)
    g = -tx.T.dot(e) / len(e)
    
    return g, e

# %% Machine Learning methods

def least_squares(y, tx):
    """ Least squares regression using normal equations. """
    
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y,tx, w)
    
    return (w, loss)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent. """
    
    w = initial_w
    for n_iter in range(max_iters):
        g, e = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * g

    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent. """
    
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            g, e = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * g
            loss = compute_mse(y, tx, w)

    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations. """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w= np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_mse(y,tx,w)
    
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD. """
    raise NotImplementedError
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD. """
    raise NotImplementedError
    return (w, loss)

# %% Polynomials, Split data

def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_data(x, y, ratio, seed=1):
    """ Split the dataset based on the split ratio."""
    
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
  
    index_tr = indices[: index_split] 
    index_te = indices[index_split:]
    
    x_tr = x[index_tr, :]
    x_te = x[index_te, :]
    y_tr = y[index_tr]
    y_te = y[index_te]
    
    return x_tr, x_te, y_tr, y_te

# %% Cross-validation 

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    te_indices = k_indices[k]
    tr_indices = [ind for split in k_indices for ind in split if ind not in te_indices] 
    x_tr = x[tr_indices, :]
    x_te = x[te_indices, :]
    y_tr = y[tr_indices]
    y_te = y[te_indices]
        
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
        
    w_tr, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
        
    loss_te = compute_mse(y_te, tx_te, w_tr)
    
    return np.mean(np.array(loss_tr)), np.mean(np.array(loss_te))

# %% Addtional methods

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
    