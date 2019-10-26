# Project 1 : Higgs Boson
Machine Learning Course (CS-433), 2019.

### Team Members

Robin Fallegger: robin.fallegger@epfl.ch

Christelle Schneuwly: christelle.schneuwly@epfl.ch

Margot Wendling: margot.wendling@epfl.ch

## Aim
The goal of this project was to be able distinguish Higgs boson signal from background in in original data from CERN using machine learning methods. 

Packages used : numpy, matplotlib.

## Files
#### `proj1_helpers.py`
Contain functions used to load the data and generate a CSV submission file.

#### `implementations.py`
Contains all the methods used in this project.

##### Exploratory Analysis
- **`plot_feature`**: Scatter plot, histogram and statistical information on a specified feature
- **`compute_feature`**: Statistics from two datasets
- **`description_feature`**: Text descrption of principal statistics
##### Pre-processing Methods
- **`missingness_filter`**: Remove features with missing data higher than a threshold
- **`impute_mean`**: Replaces missing data by feature mean
- **`impute_median`**: Replaces missing data by feature median
- **`impute_median_train`**: Replaces missing data by feature median and returns median list
- **`impute_median_from_train`**: Replaces missing data by feature median computed on train
- **`impute_gaussian`**: Replaces missing data by point in a gaussian distribution 
- **`standardize`**: Standarize features
- **`standardize_train`**: Standarize features in train data and returns mean and standard deviation
- **`standardize_test`**: Standarize features in test data using train mean and standard deviation
- **`standardize_both`**: Standarize train and test data using 2 previous functions
- **`remove_outliers`**: Remove datapoints if value is higher than a threshold
##### Data jet subsets handling
- **`separate_jet`**: Separate categorical data  based on 'jet' value
- **`merge_jet`**: Merge the predictions from different data subsetss
##### Feature processing 
- **`train_data_formatting`**: Replace missing data by impute_function and includes interaction terms
- **`build_poly`**: Compute polynomial data augmentation
- **`build_interaction`**: Compute feature interaction terms for data augmentation
- **`build_poly_inter`**: Compute polynomial data augmentation on interaction terms
##### Loss functions and Gradients
- **`sigmoid`**: Sigmoid function 
- **`compute_mse`**: Mean square error
- **`compute_loglikelihood`**: Negative loglikelihood
- **`compute_gradient`**: Gradient of least square loss function
- **`compute_log_gradient`**: Gradient of negative loglikelihood loss function 
##### Machine learning methods
- **`least_squares_gd`**: Linear regression using gradient descent
- **`least_squares_sgd`**: Linear regression using stochastic gradient descent
- **`least_squares`**: Least squares regression using normal equations
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: using stochastic gradient descent
- **`reg_logistic_regression`**: Regularized logistic regression
As well as additional methods used for code optimization. 
##### Mini-batch and Split data
- **`batch_iter`**: Mini-batch creation for stochastic gradient descent
- **`split_data`**: Split data in test and train sets
##### Cross-validation and Bias-variance Decomposition
- **`build_k_indices`**: Build indices for cross-validation
- **`cross_validation`**: Cross-validation for all methods based on loss or negative log likelihood minimization
- **`cross_accuracy_measures`**: Computation of accuracy, precision, recall and F-score
- **`cross_validation_wAcc`**: Cross-validation for all methods based on accuracy measures maximization
- **`cross_validation_visualization`**: Plots cross-validation with loss function measures
- **`bias_variance_decomposition_visualization`**: Plots biais-variance analysis
- **`vis_cv_acc`**: Plots cross-validation with accuracy measures
##### Predictions
- **`our_predict_labels`**: Label prediction with 0.5 threshold for logistic regression and regularized logistic regression
- **`build_final_model`**: Return weights once optimal parameters are set, accuracy measures and statistics of the data
- **`make prediction`**: Return predicted labels with trained weights

#### `run.py`
Script that generates the exact CSV file submitted on Kaggle.

#### `project1.ipynb`
Python notebook used for tests during this project.

## Run
To get the exact results run the `run.py` file.

## Report
Overleaf link to Report : https://www.overleaf.com/6926763958hrcwkjxpbkct
