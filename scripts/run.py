# -*- coding: utf-8 -*-
""" Implementation of Machine Learning functions. """
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from implementations import *

# %% Load training data 

y, tX, ids = load_csv_data('../data/train.csv')

# %% Pre-process training data and divide into jet subsets

tX = log_transformation(tX)

# Separate into 3 groups based on jet values
_, y_jet0, tX_jet0, _, y_jet1, tX_jet1, _, y_jet2, tX_jet2 = separate_jet(y, tX)

# Remove features with too much missing data
missingness_cutoff = 0.95
tX_jet0, rmX_jet0 = missingness_filter(tX_jet0, missingness_cutoff)
tX_jet1, rmX_jet1 = missingness_filter(tX_jet1, missingness_cutoff)
tX_jet2, rmX_jet2 = missingness_filter(tX_jet2, missingness_cutoff)
tX_jet0, median_jet0 = impute_median_train(tX_jet0)
tX_jet1, median_jet1 = impute_median_train(tX_jet1)
tX_jet2, median_jet2 = impute_median_train(tX_jet2)

# %% Build and train a model on each sub datatset

w_interaction = True

degree_ri_jet0 = 10
lambda_ri_jet0 = 3.831e-15
w0_ri, _, _, data_meas0_ri = build_final_model(y_jet0, tX_jet0, degree_ri_jet0,
                                                      lambda_ri_jet0, ml_function = 'ri', interaction = w_interaction)
degree_ri_jet1 = 13
lambda_ri_jet1 = 1e-12
w1_ri, _, _,  data_meas1_ri = build_final_model(y_jet1, tX_jet1, degree_ri_jet1,
                                                      lambda_ri_jet1, ml_function = 'ri', interaction = w_interaction)
degree_ri_jet2 = 14
lambda_ri_jet2 = 1e-13
w2_ri, _, _,  data_meas2_ri = build_final_model(y_jet2, tX_jet2, degree_ri_jet2,
                                                      lambda_ri_jet2, ml_function = 'ri', interaction = w_interaction)

# %% Load testing data

_, tX_test, ids_test = load_csv_data('../data/test.csv')

# %% Pre-process testing data and divide into jet subsets

tX_test = log_transformation(tX_test)
idx_test0, _, tX_test0, idx_test1, _, tX_test1, idx_test2, _, tX_test2 = separate_jet(np.zeros(tX_test.shape[0]), tX_test)

# %% Make predictions

y_test0 = make_prediction(tX_test0, w0_ri, rmX_jet0, median_jet0,
                          degree_ri_jet0, data_meas0_ri, interactions = w_interaction, ml_function = "ri")
y_test1 = make_prediction(tX_test1, w1_ri, rmX_jet1, median_jet1,
                          degree_ri_jet1, data_meas1_ri, interactions = w_interaction, ml_function = "ri")
y_test2 = make_prediction(tX_test2, w2_ri, rmX_jet2, median_jet2,
                          degree_ri_jet2, data_meas2_ri, interactions = w_interaction, ml_function = "ri")

# Merge the predicted labels of each jet subset
y_pred = merge_jet(idx_test0, y_test0, idx_test1, y_test1, idx_test2, y_test2)
create_csv_submission(ids_test, y_pred, '../data/final_submission.csv')
