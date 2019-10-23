# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np
import matplotlib.pyplot as plt


# %% Explosratory Analysis - and its Plots

def plot_feature(ids, tX, y, f, bins=20):
    """returns three subfirgures representing one given feature: a scatter plot of the values regarding the sample, 
    a histogram of the apparition per value, and finally statistical information about the given distributions.
    A color distinction is also shown regarding the label."""
    # inputs:
    #   - ids is the list of the index of the samples
    #   - tX is the array of samples (for which each feature has a given value)
    #   - y is the list of index (-1 or +1)
    #   - f is the number of the feature we are interessed in
    #   - bins is a parameter of the histogram
    
    print ('Feature {}:'.format(f))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot of the feature regarding the sample
    col = np.where(y==1, 'b', 'orange')
    ax[0].scatter(ids, tX[:,f], c=col)
    ax[0].set_title('Values of feature {} regarding the sample'.format(f))
    ax[0].set_xlabel('Samples')
    ax[0].set_ylabel('Values')
    
    # Histogram plot of the apprition of a given feature value
    tX_plus = []
    tX_minus = []
    for n in range(tX.shape[0]):
        if (y[n]==1):
            tX_plus.append(tX[n,f])
        elif (y[n]==-1):
            tX_minus.append(tX[n,f])
    ax[1].hist(tX_plus, bins,alpha=0.5, label='label +1', color='b')
    ax[1].hist(tX_minus, bins,alpha=0.5, label='label -1', color='orange')
    ax[1].legend(loc='upper right')
    ax[1].set_title("Histogram of factor {} distribution".format(f))
    ax[1].set_xlabel('Values of the feature {}'.format(f))
    ax[1].set_ylabel('Number of apparitions')
    
    # Table
    means, std, d, n_usable, n_tot = compute_feature(tX_minus, tX_plus)
    text = description_feature(means, std, d, n_usable, n_tot)
    ax[2].text(0.5, 0.2, text, fontsize=14, ha='center')
    plt.show()
    
    
def compute_feature(list1, list2):
    """compute some statistical information from two lists of data (of different labels) - values could be nan!"""
    # inputs: 
    #   - list1 corresponds to the data of the same first label
    #   - list2 corresponds to the data of the same second label
    # outputs: 
    #   - means is composed of [mean of both lists together, mean of list 1, mean of list 2]
    #   - std is composed of [std of both lists together, std of list 1, std of list 2]
    #   - d is the discriminability of the two distributions representing the two given lists of data
    #   - n_usable is the total number (over the two lists) of non-nan values
    #   - n_tot is the toal number (over the two lists) of values (nan or not)
    
    means = [ np.nanmean(list1+list2), np.nanmean([list1]), np.nanmean([list2]) ]
    std = [ np.nanstd(list1+list2), np.nanstd([list1]), np.nanstd([list2]) ]
    
    n1 = len(list1)
    n2 = len(list2)
    n_tot = n1+n2
    n_usable =  np.count_nonzero(~np.isnan(list1))+np.count_nonzero(~np.isnan(list2))
    
    inter_class = n1*(means[1]-means[0])**2 + n2*(means[2]-means[0])**2; 
    intra_class = (n1-1)*std[1] + (n2-1)*std[2];
    d = inter_class / intra_class
    
    return means, std, d, n_usable, n_tot


def description_feature(means, std, d, n_usable, n_tot):
    """return a text describing the principal statistical characteristics given in argument"""
    # inputs (same as the outputs of compute_feature):
    #   - means is composed of [mean of both lists together, mean of list 1, mean of list 2]
    #   - std is composed of [std of both lists together, std of list 1, std of list 2]
    #   - d is the discriminability of the two distributions representing the two given lists of data
    #   - n_usable is the total number (over the two lists) of non-nan values
    #   - n_tot is the toal number (over the two lists) of values (nan or not)
    # output:
    #   - text
    
    text = "means:" + '\n'
    text += "tot -- (-1) -- (+1)" + '\n'
    text += "{}".format(round(means[0],2)) + " -- " + "{}".format(round(means[1],2)) + " -- " + "{}".format(round(means[2],2)) + '\n' + '\n'
    text += "std:" + '\n'
    text += "tot -- (-1) -- (+1)" + '\n'
    text += "{}".format(round(std[0],2)) + " -- " + "{}".format(round(std[1],2)) + " -- " + "{}".format(round(std[2],2)) + '\n' + '\n'
    text += "d:" + '\n'
    text += "{}".format(d) + '\n' + '\n'
    text += "usable n:"+ '\n'
    text += "{}/{} = {}%".format(n_usable, n_tot, round(100*n_usable/n_tot))
    return text


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

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(degrees, rmse_tr.T, 'b', linestyle="-", color=([0.7, 0.7, 1]),
        label='train', linewidth=0.3)
    plt.plot(degrees, rmse_te.T, 'r', linestyle="-", color=[1, 0.7, 0.7], label='test', linewidth=0.3)
    plt.plot(degrees, rmse_tr_mean.T, 'b', linestyle="-", label='train', linewidth=3)
    plt.plot(degrees, rmse_te_mean.T, 'r', linestyle="-", label='test', linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    