# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

# %% Exploratory Analysis

def plot_feature(ids, tX, y, f, bins=20):
    """ Returns three subfirgures representing one given feature: a scatter plot of the values regarding the sample,
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
    """ Compute some statistical information from two lists of data (of different labels) - values could be nan!"""
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
    """ Return a text describing the principal statistical characteristics given in argument."""
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

# %% Pre-processing Methods

def missingness_filter(cX, cutoff = 0.5):
    """ Removes all features with more than the missingness cutoff and all-zero features. """

    cX = np.where(cX == -999, np.nan, cX)
    missingness = np.sum(np.isnan(cX), axis = 0)/cX.shape[0]
    # keep index of features with missing data proportion higher than the threshold
    to_remove = np.where(missingness > cutoff)[0]
    # keep index with all-zero features
    zero_col = np.where((cX==0).all(0))
    # combine indices of all the features to remove
    to_remove= np.append(to_remove, zero_col)

    return np.delete(cX, to_remove, axis = 1), to_remove


def impute_mean(x):
    """ Replaces missing datapoints in x by the mean value of non missing data."""

    #compute mean of all features, ignoring nan/missing values
    mean = np.nanmean(x, axis=0)
    #find indices of missing data
    inds = np.where(np.isnan(x))
    #replace by the mean of the feature
    x[inds] = np.take(mean, inds[1])

    return x


def impute_median(x):
    """ Replaces missing datapoints in x by the median of non missing data."""

    #compute median of all features, ignoring nan/missing values
    median = np.nanmedian(x, axis=0)
    #find indices of missing data
    inds = np.where(np.isnan(x))
    #replace by the median of the feature
    x[inds] = np.take(median, inds[1])

    return x


def impute_median_train(x):
    """ Replaces missing datapoints in x by the median of non missing data.
    Also returns the vector of medians, used to impute the missing values of the test data"""

    #compute median of all features, ignoring nan/missing values
    median = np.nanmedian(x, axis=0)
    #find indices of missing data
    inds = np.where(np.isnan(x))
    #replace by the median of the feature
    x[inds] = np.take(median, inds[1])

    return x, median


def impute_median_from_train(x, median):
    """ Replaces missing datapoints in x by the median of non missing data.
    Used for imputing the missing data of the test set from the corresponding median
    of the train set"""

    #find indices of missing data
    inds = np.where(np.isnan(x))
    #replace by the median of the feature given as input
    x[inds] = np.take(median, inds[1])

    return x


def impute_gaussian(x):
    """ Replaces missing datapoints in x by a random value in a gaussian distribution."""

    #find indices of missing data
    inds = np.where(np.isnan(x))
    #compute mean, ignoring nan/missing values
    mean = np.nanmean(x, axis=0)
    #compute standard deviation, ignoring nan/missing values
    std = np.nanstd(x, axis=0)
    #replace by the mean and some Gaussian noise
    x[inds] = np.take(np.random.normal(loc=mean, scale=std), inds[1])

    return x


def standardize(x):
    """ Standardize the original data set."""

    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def standardize_train(x):
    """Standardize features in train data and returns mean and standard deviation"""

    #compute the mean of the columns
    mean_x = np.mean(x, axis = 0)
    #compute the standard deviation of the columns
    std_x = np.std(x, axis = 0)
    #replace 0 stds in order to prevent division by 0 errors
    std_x = np.where(std_x == 0, 1, std_x)
    #center the values
    x = x - mean_x
    #scale the values
    x = x / std_x

    #return the mean and std, for standardizing the test set
    return x, mean_x, std_x


def standardize_test(x, tr_mean, tr_std):
    """Standardize features in test data using train mean and standard deviation"""

    x = x - tr_mean
    x = x / tr_std

    return x


def standardize_both(x_train, x_test):
    """Standardize train and test data using 2 previous functions"""

    std_train_x, mean_x, std_x = standardize_train(x_train)
    std_test_x = standardize_test(x_test, mean_x, std_x)

    #return standardized data sets
    return std_train_x, std_test_x


def remove_outliers(y, x, feature_index, threshold):
    """ Removes in y and x the data points if for a list of features,
    the data point is higher the associated threshold."""

    for f_idx, thres in zip(feature_index, threshold):
        indices = [i for (i, xi) in enumerate(x[:, f_idx]) if xi > thres]
        x = np.delete(x, indices, axis=0)
        y = np.delete(y, indices, axis=0)

    return y, x

# %% Data jet subsets handling

def separate_jet(y, tx):
    """ Separate the dataset based on their categorial feature 'jet'."""

    idx0 = np.where(tx[:, 22] == 0)
    idx1 = np.where(tx[:, 22] == 1)
    idx2 = np.where(tx[:, 22] >= 2)

    y_jet0 = y[idx0]
    tx_jet0 = tx[idx0]
    y_jet1 = y[idx1]
    tx_jet1 = tx[idx1]
    y_jet2 = y[idx2]
    tx_jet2 = tx[idx2]

    # remove categorical data column
    tx_jet0 = np.delete(tx_jet0, 22, axis=1)
    tx_jet1 = np.delete(tx_jet1, 22, axis=1)
    tx_jet2 = np.delete(tx_jet2, 22, axis=1)

    return idx0, y_jet0, tx_jet0, idx1, y_jet1, tx_jet1, idx2, y_jet2, tx_jet2


def merge_jet(idx0, y_pred0, idx1, y_pred1, idx2, y_pred2):
    """ Merge the predictions generated using the separated weights generated by training on sub-datasets."""

    y = np.empty(len(y_pred0)+len(y_pred1)+len(y_pred2))
    y[idx0]=y_pred0
    y[idx1]=y_pred1
    y[idx2]=y_pred2

    return y

# %% Feature processing

def build_poly(x, d):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree."""

    poly = np.ones((len(x), 1))
    for deg in range(1, d+1):
        n_poly = np.power(x, deg)
        #n_poly = np.apply_along_axis(standardize, 1, n_poly)
        poly = np.c_[poly, n_poly]

    return poly


def build_interaction(x):
    """ Build a matrix containing the interaction terms of the input features e.g. x1 * x2, x1 * x3, etc """

    #column of ones
    comb = np.ones((x.shape[0],)).reshape(-1,1)

    #the way the looping is done prevents having duplicate columns (x1*x2 and x2*x1)
    #and squares of columns (x2*x2) for example

    #loop on the columns, ignoring the last one
    for i in range(x.shape[1]-1):
        #loop on the columns above i, up to the last one
        for j in range(i+1,x.shape[1]):
            #product of two columns
            temp = x[:,i] * x[:,j]

            #add to matrix
            comb = np.concatenate((comb, temp.reshape(-1,1)), axis=1)

    #return the matrix after removing the column containing only ones
    return np.delete(comb, 0, axis = 1)


def build_poly_inter(x, d, interaction = False):
    """Compute polynomial data augmentation with or without interaction terms"""

    #number of columns/features of the input matrix x
    n_features = x.shape[1]

    #column of ones (offset)
    comb = np.ones((x.shape[0],)).reshape(-1,1)

    #build the polynomial matrix, remove the first column containing only ones (offset)
    poly = np.delete(build_poly(x, d), 0, axis = 1)

    if interaction:
        #build first order interactions
        inter = build_interaction(x)
        #combine the interaction and polynomial matrix
        poly = np.concatenate((inter, poly), axis=1)

    #add the offset at the front of the data matrix
    x = np.concatenate((comb, poly), axis=1)

    return x


# %% Loss functions and Gradients

def sigmoid(t):
    """ Apply sigmoid function on t."""

    return 1/(1+np.exp(-t))


def compute_mse(y, tx, w):
    """ Compute the mse for vector e."""

    e = y - tx.dot(w)

    return e.T.dot(e) / (2*len(y))


def compute_loglikelihood(y, tx, w):
    """ Compute the cost by negative log likelihood."""

    h = sigmoid(tx.dot(w))

    return -y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h))


def compute_gradient(y, tx, w):
    """ Compute the gradient."""

    e = y - tx.dot(w)

    return - tx.T.dot(e) / len(e)


def compute_log_gradient(y, tx, w):
    """ Compute the gradient of loss."""

    h = sigmoid(tx.dot(w))

    return tx.T.dot(h-y)

# %% Machine Learning Methods

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
    """ Logistic regression using GD. """

    w = initial_w
    for n_iter in range(max_iters):
        g = compute_log_gradient(y, tx, w)
        w = w - gamma * g
        loss = compute_loglikelihood(y, tx, w)

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using GD. """

    N = len(y)
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_log_gradient(y, tx, w)+lambda_*w/N
        w = w - gamma * g
        loss = compute_loglikelihood(y, tx, w)+lambda_*np.sum(w**2)/(2*N)

    return (w, loss)

# %% Mini-batch and Split data

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


def split_data(x, y, ratio, seed=1):
    """ Split the dataset based on the split ratio."""

    np.random.seed(seed)
    num_row = len(y)

    #make random permutation of the row indices
    indices = np.random.permutation(num_row)

    #assign indices to train or test
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    #split into test and train
    x_tr = x[index_tr, :]
    x_te = x[index_te, :]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te

# %% Cross-validation and Bias-variance Decomposition

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold."""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def compute_accuracy_measures(true_labels, pred_labels):
    """Computes accuracy measures from the true and predicted labels
    if you only want one of the following measures, you can just put a _ at the corresponding outputs

    Returns:
    --------
    accuracy: scalar
        accuracy of the model (tp + tn / total number of samples)
    precision: scalar
        precision of the model (tp/(tp + fp))
    recall: scalar
        recall of the model (tp/(tp + fn))
    F1: scalar
        model's F1 measure (2*precision*recall/(precision + recall))
    """

    if true_labels.shape[0] != pred_labels.shape[0]:
        print("Labels given are not of the same shape !")
        print("True labels shape:", true_labels.shape)
        print("Predicted labels shape:", pred_labels.shape)
        return

    #initialize counters
    tp = 0 #true positives
    fn = 0 #false negatives
    fp = 0 #false positives
    correct = 0 #number of correct predictions

    for true, pred in zip(true_labels, pred_labels):
        if true == 1 and pred == 1:
            tp = tp + 1 #true positives
        elif true == 1 and pred == -1:
            fn = fn + 1 #false negatives
        elif true == -1 and pred == 1:
            fp = fp + 1 #false positives
        if true == pred:
            correct = correct + 1

    accuracy = correct / true_labels.shape[0]

    if tp + fp != 0:
        precision = tp/(tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp/(tp + fn)
    else:
        recall = 0

    if (precision + recall) != 0:
        F1 = 2*precision*recall/(precision + recall)
    else:
        F1 = 0

    return accuracy, precision, recall, F1


def cross_validation_wAcc(y, x, k_indices, k_fold, degrees, lambdas = [0], ml_function = 'ls', max_iters = 0, gamma = 0.05, verbose = False, interaction = False):
    """ Returns a list the train losses and test losses of the cross validation."""

    #build data matrix with max number of degrees
    n_features = x.shape[1]
    comb = np.ones((x.shape[0],)).reshape(-1,1)
    poly = np.delete(build_poly(x, degrees[-1]), 0, axis = 1)
    if interaction:
        inter = build_interaction(x)
        print("Added {} interaction features" .format(inter.shape[1]))
        poly = np.concatenate((inter, poly), axis=1)
    x = np.concatenate((comb, poly), axis=1)

    print("Finished preparing data for cross-validation \n")

    #initialize matrices of loss and accuracy measures (accuracy, precision, recall and f1)
    losses_tr_cv = np.empty((len(lambdas), len(degrees)))
    losses_te_cv = np.empty((len(lambdas), len(degrees)))
    acc_tr_cv = np.empty((len(lambdas), len(degrees)))
    acc_te_cv = np.empty((len(lambdas), len(degrees)))
    pre_tr_cv = np.empty((len(lambdas), len(degrees)))
    pre_te_cv = np.empty((len(lambdas), len(degrees)))
    rec_tr_cv = np.empty((len(lambdas), len(degrees)))
    rec_te_cv = np.empty((len(lambdas), len(degrees)))
    f1_tr_cv = np.empty((len(lambdas), len(degrees)))
    f1_te_cv = np.empty((len(lambdas), len(degrees)))

    #loop over lambdas
    for index_lambda, lambda_ in enumerate(lambdas):

        #loop over degrees
        for index_degree, degree in enumerate(degrees):
            #initialize matrices of loss and accuracy measures (accuracy, precision, recall and f1)
            #for the k-folds
            losses_tr = np.empty(k_fold)
            losses_te = np.empty(k_fold)
            accuracies_tr = np.empty(k_fold)
            accuracies_te = np.empty(k_fold)
            precisions_tr = np.empty(k_fold)
            precisions_te = np.empty(k_fold)
            recalls_tr = np.empty(k_fold)
            recalls_te = np.empty(k_fold)
            F1s_tr = np.empty(k_fold)
            F1s_te = np.empty(k_fold)

            #loop over k folds
            for k in range(k_fold):
                loss_tr = 0
                loss_te = 0

                #make the train/test split for this fold
                te_indices = k_indices[k]
                tr_indices = [ind for split in k_indices for ind in split if ind not in te_indices]

                #return the index of the last column to be used for this degree
                if interaction:
                    degree_idx = (degree * n_features) + inter.shape[1] + 1
                else:
                    degree_idx = degree * n_features + 1

                #split into train and test, and select the correct number of columns
                #for this degree
                x_tr = x[tr_indices, 0:degree_idx].copy()
                x_te = x[te_indices, 0:degree_idx].copy()
                y_tr = y[tr_indices]
                y_te = y[te_indices]

                #standardize the train and test sets
                tx_tr, tx_te = standardize_both(x_tr, x_te)


                #fit the selected model

                if ml_function == 'gd':
                    initial_w = np.zeros(tx_tr.shape[1])
                    w_tr, loss_tr = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
                    loss_te = compute_mse(y_te, tx_te, w_tr)

                if ml_function == 'sgd' :
                    initial_w = np.zeros(tx_tr.shape[1])
                    w_tr, loss_tr = least_squares_SGD(y_tr, tx_tr, initial_w, max_iters, gamma)
                    loss_te = compute_mse(y_te, tx_te, w_tr)

                if ml_function == 'ri':
                    w_tr, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
                    loss_te = compute_mse(y_te, tx_te, w_tr)

                if ml_function == 'lr':
                    initial_w = np.zeros(tx_tr.shape[1])
                    w_tr, loss_tr = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
                    loss_te = compute_loglikelihood(y_te, tx_te, w_tr)

                if ml_function == 'rlr':
                    initial_w = np.zeros(tx_tr.shape[1])
                    w_tr, loss_tr = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
                    loss_te = compute_loglikelihood(y_te, tx_te, w_tr)

                #save the loss for this fold
                losses_tr[k] = loss_tr
                losses_te[k] = loss_te

                if np.isnan(w_tr).any():
                    print(np.sum(np.isnan(w_tr), axis = 0)/w_tr.shape[0], " weights are nan \n")

                #make prediction for the test set
                if ml_function == 'lr' or ml_function == 'rlr':
                    y_tr_pred = our_predict_labels(w_tr, tx_tr, True)
                    y_te_pred = our_predict_labels(w_tr, tx_te, True)
                else:
                    y_tr_pred = our_predict_labels(w_tr, tx_tr)
                    y_te_pred = our_predict_labels(w_tr, tx_te)

                #compute the accuracy measures for the predictions on the test set in this fold
                accuracies_tr[k], precisions_tr[k], recalls_tr[k], F1s_tr[k] = compute_accuracy_measures(y_tr, y_tr_pred)
                accuracies_te[k], precisions_te[k], recalls_te[k], F1s_te[k] = compute_accuracy_measures(y_te, y_te_pred)

            #make the mean of the loss over the folds
            if ml_function == 'gd' or 'sgd' or 'ri':
                losses_tr_cv[index_lambda][index_degree] = np.mean(np.sqrt(2*losses_tr))
                losses_te_cv[index_lambda][index_degree] = np.mean(np.sqrt(2*losses_te))

            if ml_function == 'lr' or 'rlr':
                losses_tr_cv[index_lambda][index_degree] = np.mean(losses_tr)
                losses_te_cv[index_lambda][index_degree] = np.mean(losses_te)

            #make the mean of the accuracy measures over the folds
            acc_tr_cv[index_lambda][index_degree] = np.mean(accuracies_tr)
            acc_te_cv[index_lambda][index_degree] = np.mean(accuracies_te)
            pre_tr_cv[index_lambda][index_degree] = np.mean(precisions_tr)
            pre_te_cv[index_lambda][index_degree] = np.mean(precisions_te)
            rec_tr_cv[index_lambda][index_degree] = np.mean(recalls_tr)
            rec_te_cv[index_lambda][index_degree] = np.mean(recalls_te)
            f1_tr_cv[index_lambda][index_degree] = np.mean(F1s_tr)
            f1_te_cv[index_lambda][index_degree] = np.mean(F1s_te)

            if verbose == True:
                print('Completed degree '+str(index_degree+1)+'/'+str(len(degrees)),end="\r",flush=True)

        if verbose == True:
            print('\n Completed lambda '+str(index_lambda+1)+'/'+str(len(lambdas)) + '\n',end="\r",flush=True)

    #make dictionary of the accuracy measures
    acc_measures = {"acc_tr": acc_tr_cv, "acc_te": acc_te_cv, "pre_tr": pre_tr_cv, "pre_te": pre_te_cv,
        "rec_tr": rec_tr_cv, "rec_te": rec_te_cv, "f1_tr": f1_tr_cv, "f1_te": f1_te_cv}

    return losses_tr_cv, losses_te_cv, acc_measures


def cross_validation_visualization(degrees, loss_tr, loss_te, lambds=[], y_label = "error"):
    """visualization the curves of train error and test error."""
    N = len(degrees)
    cmap = plt.get_cmap('jet_r')
    mask_tr = np.isfinite(loss_tr)
    mask_te = np.isfinite(loss_te)
    if np.array(loss_tr).shape[0] > 1 :
        for index_degree, degree in enumerate(degrees):
            color = cmap(float(index_degree)/N)
            plt.semilogx(lambds[mask_tr[:, index_degree]], loss_tr[:, index_degree][mask_tr[:, index_degree]],
                         marker=".", linewidth = 0.5, color = color, label='deg'+str(degree))
            plt.semilogx(lambds[mask_te[:, index_degree]], loss_te[:, index_degree][mask_te[:, index_degree]],
                         marker="*", linewidth = 0.5, color = color,  label='deg'+str(degree))
        plt.xlabel("lambda")
    if np.array(loss_tr).shape[0] == 1 :
        plt.plot(np.array(degrees)[mask_tr.flatten()],
                     loss_tr[mask_tr], marker=".", linewidth = 0.5, color='b', label='train')
        plt.plot(np.array(degrees)[mask_te.flatten()],
                     loss_te[mask_te], marker="*", linewidth = 0.5, color='r',  label='test')
        plt.xlabel("degree")
    plt.ylabel(y_label)
    plt.title("cross validation")
    plt.legend(loc=1)
    plt.grid(True)


def bias_variance_decomposition_visualization(degrees, loss_tr, loss_te):
    """ Visualize the bias variance decomposition."""

    loss_tr_mean = np.expand_dims(np.mean(loss_tr, axis=0), axis=0)
    loss_te_mean = np.expand_dims(np.mean(loss_te, axis=0), axis=0)
    plt.plot(degrees, loss_tr.T, 'b', linestyle="-", label='train', linewidth=0.3)
    plt.plot(degrees, loss_te.T, 'r', linestyle="-", label='test', linewidth=0.3)
    plt.plot(degrees, loss_tr_mean.T, 'b', linestyle="-", label='train', linewidth=3)
    plt.plot(degrees, loss_te_mean.T, 'r', linestyle="-", label='test', linewidth=3)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.ylim(0, 10)
    plt.title("Bias-Variance Decomposition")


def vis_cv_acc(degrees,lambdas,acc_measures):
    """Plots accuracy measures from the cross-validation"""
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    cross_validation_visualization(degrees, acc_measures["acc_tr"], acc_measures["acc_te"], lambdas, "accuracy")
    plt.subplot(2, 2, 2)
    cross_validation_visualization(degrees, acc_measures["pre_tr"], acc_measures["pre_te"], lambdas, "precision")
    plt.subplot(2, 2, 3)
    cross_validation_visualization(degrees, acc_measures["rec_tr"], acc_measures["rec_te"], lambdas, "recall")
    plt.subplot(2, 2, 4)
    cross_validation_visualization(degrees, acc_measures["f1_tr"], acc_measures["f1_te"], lambdas, "F1")

# %% Predictions

def our_predict_labels(weights, data, log = False):
    """Generates class predictions given weights, and a test data matrix"""

    if log:
        y_pred = np.dot(data, weights)
        y_pred[np.where(y_pred <= 0.5)] = -1
        y_pred[np.where(y_pred > 0.5)] = 1

    else:
        y_pred = predict_labels(weights, data)

    return y_pred


def build_final_model(y, x, degree, lambda_, ml_function = 'ri', gamma = 0.05, interaction = False):
    """Return weights once optimal parameters are set, accuracy measures and statistics of the data"""

    #build the data matrix
    x = build_poly_inter(x, degree, interaction)
    #standardize the train set
    tx, mean_x, std_x = standardize_train(x)

    #output the mean and std for each column, to be used to standardize the test set
    data_measures = {"mean": mean_x,
                     "std": std_x}

    #fit the correct model
    if ml_function == 'gd':
        initial_w = np.zeros(tx.shape[1])
        w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)

    if ml_function == 'sgd' :
        initial_w = np.zeros(tx.shape[1])
        w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)

    if ml_function == 'ri':
        w, loss = ridge_regression(y, tx, lambda_)

    if ml_function == 'lr':
        initial_w = np.zeros(tx.shape[1])
        w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

    if ml_function == 'rlr':
        initial_w = np.zeros(tx.shape[1])
        w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

    #make predictions on the train set
    if ml_function == 'lr' or ml_function == 'rlr':
        y_pred = our_predict_labels(w, tx, True)
    else:
        y_pred = our_predict_labels(w, tx)

    #compute accuracy measures of the complete model on the train set
    accuracy, precision, recall, F1 = compute_accuracy_measures(y, y_pred)
    acc_measures = {"acc": accuracy, "pre": precision, "rec": recall, "f1": F1}

    return w, loss, acc_measures, data_measures


def make_prediction(tx, weights, rmx, median, d, train_data_measures, interactions = False, ml_function = "ri"):
    """Return predicted labels with trained weights"""

    #delete features
    print("Deleting features...",end="")
    tx = np.delete(tx, rmx, axis = 1)
    print("done",end="\n")

    #impute missing data
    print("Imputing missing data...",end="")
    tx = impute_median_from_train(tx, median)
    print("done",end="\n")

    #data augmentation
    print("Building data matrix...",end="")
    tx = build_poly_inter(tx, d, interactions)
    print("done",end="\n")

    assert tx.shape[1] == weights.shape[0]

    #standardization
    print("Standardizing data matrix...",end="")
    tx = standardize_test(tx, train_data_measures["mean"], train_data_measures["std"])
    print("done",end="\n")

    print("Making predictions...",end="")
    if ml_function == "lr" or ml_function == "rlr":
        y_pred = our_predict_labels(weights, tx, log = True)
    else:
        y_pred = our_predict_labels(weights, tx, log = False)
    print("done")

    return y_pred
