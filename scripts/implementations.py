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

# %% Loss functions, Gradients and Hessians

def sigmoid(t):
    """apply sigmoid function on t."""

    return 1/(1+np.exp(-t))


def compute_mse(y, tx, w):
    """ Calculate the mse for vector e."""
    
    e = y - tx.dot(w)
    return e.T.dot(e) / (2*len(y))


def compute_loglikelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    h = sigmoid(tx.dot(w))
    return -y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h))


def compute_gradient(y, tx, w):
    """ Compute the gradient."""
    
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


def compute_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    
    h = sigmoid(tx.dot(w))
    return tx.T.dot(h-y)


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
        g = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * g
        
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent. """
    
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            g = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * g
            loss = compute_mse(y, tx, w)
            
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations. """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_mse(y,tx,w)
    
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using SGD. """
    
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_log_gradient(y, tx, w)
        w = w - gamma * g
        loss = compute_loglikelihood(y, tx, w)
            
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using SGD. """
    
    N = len(y)
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_log_gradient(y, tx, w)+lambda_*w/N
        w = w - gamma * g
        loss = compute_loglikelihood(y, tx, w)+lambda_*np.sum(w**2)/(2*N)
            
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

def cross_validation_log(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    te_indices = k_indices[k]
    tr_indices = [ind for split in k_indices for ind in split if ind not in te_indices] 
    x_tr = x[tr_indices, :]
    x_te = x[te_indices, :]
    y_tr = y[tr_indices]
    y_te = y[te_indices]
        
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
        
    w_tr, loss_tr = reg_logistic_regression(y_tr, tx_tr, lambda_)
        
    loss_te = compute_loglikelihood(y_te, tx_te, w_tr)
    
    return np.mean(np.array(loss_tr)), np.mean(np.array(loss_te))

# %% Addtional methods

def separate_factor(x, nlevels=4, column_idx = 22):
    """ Transform a column with categorical variables into different columns with binary data.
    E.g feature = [1, 2, 1, 3] -> features = [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]  """
    
    new_var = np.zeros((x.shape[0], nlevels))

    for i in range(x.shape[0]):
        for j in range(nlevels):
            if x[i,column_idx] == j:
                new_var[j,0] = 1

    x = np.delete(x, column_idx, axis = 1)

    return x, new_var


def build_interaction(x):
    """ Build a matrix containing the interaction terms of the input features e.g. x1 * x2, x1 * x3, etc """

    comb = np.ones((x.shape[0],)).reshape(-1,1)

    for i in range(x.shape[1]-1):
        for j in range(i+1,x.shape[1]):
            temp = x[:,i] * x[:,j]
            comb = np.concatenate((comb, temp.reshape(-1,1)), axis=1)
            
    return np.delete(comb, 0, axis = 1)


def missingness_filter(cX, cutoff = 0.5):
    """ Removes all features with more than the missingness cutoff """

    cX = np.where(cX == -999, np.nan, cX)
    missingness = np.sum(np.isnan(cX), axis = 0)/cX.shape[0]

    to_remove = np.where(missingness > cutoff)[0]

    return np.delete(cX, to_remove, axis = 1), to_remove


def impute_mean(x):
    
    out = np.zeros(x.shape)
    for i in range(x.shape[1]):
        temp = x[:,i]
        mean = np.nanmean(temp)
        out[:,i] = np.nan_to_num(temp, nan = mean)

    return out


def impute_median(x):
    
    out = np.zeros(x.shape)
    for i in range(x.shape[1]):
        temp = x[:,i]
        median = np.nanmedian(temp)
        out[:,i] = np.nan_to_num(temp, nan = median)

    return out
    
    
def impute_gaussian(x):
    out = np.zeros(x.shape)
    for i in range(x.shape[1]):
        temp = x[:,i]
        mean = np.nanmean(temp)
        std = np.nanstd(temp)

        for j in range(x.shape[0]):
            out[j,i] = np.nan_to_num(temp[j], nan = np.random.normal(loc=mean, scale=std))

    return out

def train_data_formatting(tX, degree = 2, cutoff = 0.6, imputation = impute_mean, interaction = False):

    #separating out the categorical variables
    cont_X, fac_X = separate_factor(tX)
    #applying a missingness filter on the columns/features
    cont_X, to_remove = missingness_filter(cont_X, cutoff)
    #imputing the missing data
    cont_X = imputation(cont_X)
    poly = build_poly(cont_X, degree)
    poly = np.concatenate((poly, fac_X), axis=1)
    
    if lin_comb:
        inter = build_interaction(cont_X)
        return np.concatenate((poly, inter), axis=1), to_remove

    return poly, to_remove