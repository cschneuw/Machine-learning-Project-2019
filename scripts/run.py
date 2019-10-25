# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from implementations import *

# %% Load training data 

y, tX, ids = load_csv_data('../data/train.csv')

# %% Pre-process training data and divide into jet subsets

y, tX = remove_outliers(y, tX, [0, 2, 3, 8, 13, 16, 19, 21, 23, 26],
                       [1100, 1000, 1000, 2500, 500, 500, 800, 1800, 800, 600])

# Separate into 3 groups based on jet values
_, y_jet0, tX_jet0, _, y_jet1, tX_jet1, _, y_jet2, tX_jet2 = separate_jet(y, tX)

# Remove features with too much missing data
tX_jet0, rmX_jet0 = train_data_formatting(tX_jet0, degree = 1, cutoff = 0.95, 
                      imputation = impute_median, interaction = False)
tX_jet1, rmX_jet1 = train_data_formatting(tX_jet1, degree = 1, cutoff = 0.95, 
                      imputation = impute_median, interaction = False)
tX_jet2, rmX_jet2 = train_data_formatting(tX_jet2, degree = 1, cutoff = 0.95, 
                      imputation = impute_median, interaction = False)

# Standardize all the remaining feature columns
tX_jet0 = np.apply_along_axis(standardize, 1, tX_jet0)
tX_jet1 = np.apply_along_axis(standardize, 1, tX_jet1)
tX_jet2 = np.apply_along_axis(standardize, 1, tX_jet2)

# %% Build and train a model on each sub datatset

degree_jet0 = 11
lambda_jet0 = 1.6681005372000556e-08
tX0_poly = build_poly(tX_jet0, degree_jet0)    
w0_ri, mse0_ri = ridge_regression(y_jet0, tX0_poly, lambda_jet0)

degree_jet1 = 8
lambda_jet1 = 4.6415888336127725e-12
tX1_poly = build_poly(tX_jet1, degree_jet1)    
w1_ri, _ = ridge_regression(y_jet1, tX1_poly, lambda_jet1)

degree_jet2 = 11
lambda_jet2 = 2.154434690031878e-09
tX2_poly = build_poly(tX_jet2, degree_jet2)    
w2_ri, _ = ridge_regression(y_jet2, tX2_poly, lambda_jet2)

# %% Load testing data 

_, tX_test, ids_test = load_csv_data('../data/test.csv')

# %% Pre-process testing data and divide into jet subsets

idx_test0, _, tX_test0, idx_test1, _, tX_test1, idx_test2, _, tX_test2 = separate_jet(np.zeros(tX_test.shape[0]), tX_test)

# Apply pre-processing to dataset jet0
tX_test0 = np.delete(tX_test0, rmX_jet0, axis=1)
tX_test0, _ = train_data_formatting(tX_test0, degree = 1, cutoff = 1.0, imputation = impute_median, interaction = False)
tX_test0[np.isnan(tX_test0)] = 0
tX_test0 = np.apply_along_axis(standardize, 1, tX_test0)
tX_test0_poly = build_poly(tX_test0, degree_jet0)

# Apply pre-processing to dataset jet1
tX_test1 = np.delete(tX_test1, rmX_jet1, axis=1)
tX_test1, _ = train_data_formatting(tX_test1, degree = 1, cutoff = 1.0, imputation = impute_median, interaction = False)
tX_test1 = np.apply_along_axis(standardize, 1, tX_test1)
tX_test1_poly = build_poly(tX_test1, degree_jet1)    

# Apply pre-processing to dataset jet2
tX_test2 = np.delete(tX_test2, rmX_jet2, axis=1)
tX_test2, _ = train_data_formatting(tX_test2, degree = 1, cutoff = 1.0, imputation = impute_median, interaction = False)
tX_test2 = np.apply_along_axis(standardize, 1, tX_test2)
tX_test2_poly = build_poly(tX_test2, degree_jet2)

# %% Predict test labels and create submissions file

y_test0 = predict_labels(w0_ri, tX_test0_poly)
y_test1 = predict_labels(w1_ri, tX_test1_poly)
y_test2 = predict_labels(w2_ri, tX_test2_poly)

# Merge the predicted labels of each jet subset
y_pred = merge_jet(idx_test0, y_test0, idx_test1, y_test1, idx_test2, y_test2)
create_csv_submission(ids_test, y_pred, '../data/final_submission.csv')
