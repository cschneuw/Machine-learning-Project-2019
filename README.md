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
- **`log_transformation`**: Apply log transform log(1+x) to right-skewed features.
- **`missingness_filter`**: Remove features with missing data higher than a threshold
- **`impute_mean`**: Replaces missing data by feature mean
- **`impute_median`**: Replaces missing data by feature median
- **`impute_median_train`**: Replaces missing data by feature median and returns median list
- **`impute_median_from_train`**: Replaces missing data by feature median computed on train
- **`impute_gaussian`**: Replaces missing data by point in a gaussian distribution
- **`standardize_train`**: Standardize features in train data and returns mean and standard deviation
- **`standardize_test`**: Standardize features in test data using train mean and standard deviation
- **`standardize_both`**: Standardize train and test data using 2 previous functions
##### Data jet subsets handling
- **`separate_jet`**: Separate categorical data  based on 'jet' value
- **`merge_jet`**: Merge the predictions from different data subsetss
##### Feature processing
- **`build_poly`**: Compute polynomial data augmentation
- **`build_interaction`**: Compute feature interaction terms for data augmentation
- **`build_poly_inter`**: Compute polynomial data augmentation with or without interaction terms
##### Loss functions and Gradients
- **`sigmoid`**: Sigmoid function
- **`compute_mse`**: Mean square error
- **`compute_loglikelihood`**: Negative loglikelihood
- **`compute_gradient`**: Gradient of least square loss function
- **`compute_log_gradient`**: Gradient of negative loglikelihood loss function
##### Machine learning methods
- **`least_squares`**: Least squares regression using normal equations
- **`least_squares_GD`**: Linear regression using gradient descent
- **`least_squares_SGD`**: Linear regression using stochastic gradient descent
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: Logistic regression using gradient descent
- **`reg_logistic_regression`**: Regularized logistic regression using gradient descent
##### Mini-batch and Split data
- **`batch_iter`**: Mini-batch creation for stochastic gradient descent
- **`split_data`**: Split data in test and train sets
##### Cross-validation and Bias-variance Decomposition
- **`build_k_indices`**: Build indices for cross-validation
- **`compute_accuracy_measures`**: Computation of accuracy, precision, recall and F-score
- **`cross_validation_wAcc`**: Cross-validation for all methods based on accuracy measures maximization
- **`cross_validation_visualization`**: Plots cross-validation with loss function measures
- **`bias_variance_decomposition_visualization`**: Plots biais-variance analysis
- **`vis_cv_acc`**: Plots cross-validation with accuracy measures
##### Predictions
- **`our_predict_labels`**: Label prediction with 0.5 threshold for logistic regression and regularized logistic regression
- **`build_final_model`**: Return weights once optimal parameters are set, accuracy measures and statistics of the data
- **`make prediction`**: Return predicted labels with trained weights

#### `run.py`
  Script that generates the exact CSV file submitted on AIcrowd. Uses the following funtions: 
  
  `load_csv_data` to load the raw CERN data as labels y, data tX and indices ids. `separate_jet` to separate y and tX into 3 jet subsets: jet = 0, jet = 1 and jet = 2, 3. `missingness_filter` removes features with too much invalid data and `impute_median_train`replaces the remaining missing data points with feature median. We apply the same functions to the test data afterwards. 
  
 `w_interaction = True` is a boolean used to build the polynomial basis models with interaction terms and `build_final_model` creates the weight matrix and returns some descriptive statistics used to standardize the test data in `make_prediction`. 

#### `project1.ipynb`
Python notebook used for tests during this project.

## Run
To get the exact results run the `run.py` file.

## Report
Overleaf link to Report : https://www.overleaf.com/6926763958hrcwkjxpbkct
