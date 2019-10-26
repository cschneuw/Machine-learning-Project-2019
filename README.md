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
##### Pre-processing
- **`missingness_filter`**: Remove features with missing data higher than a threshold
- **`impute_mean`**: Compute feature mean
- **`impute_median`**: Compute feature media
- **`impute_gaussian`**: Compute datapoints in a gaussian distribution
- **`train_data_formatting`**: Replace missing data by impute_function 
- **`standardize`**: Standarize features
- **`remove_outliers`**: Remove datapoints if value is higher than a threshold
##### Data jet subsets handling
- **`separate_jet`**: Separate categorical data  based on 'jet' value
- **`merge_jet`**: Merge the predictions from different data subsetss
##### Loss functions, Gradients
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
##### Mini-batch, Polynomials, Split data
- **`batch_iter`**: Mini-batch creation for stochastic gradient descent
- **`build_poly`**: Compute polynomial data augmentation
- **`build_interaction`**: Compute feature interaction terms for data augmentation
- **`split_data`**: Split data in test and train sets
##### Cross-validation and Bias-variance Decomposition
- **`build_k_indices`**: Build indices for cross-validation
- **`cross_validation`**: Computes cross-validation for all methods 
- **`cross_validation_visualization`**: Plots cross-validation results
- **`bias_variance_decomposition_visualization`**: Plots biais-variance analysis

#### `run.py`
Script that generates the exact CSV file submitted on Kaggle.

#### `project1.ipynb`
Python notebook used for tests during this project.

## Run
To get the exact results run the `run.py` file.

## Report
Overleaf link to Report : https://www.overleaf.com/6926763958hrcwkjxpbkct
