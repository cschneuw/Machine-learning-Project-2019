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
Contain the 6 regression methods needed for this project
- **`least_squares_gd`**: Linear regression using gradient descent
- **`least_squares_sgd`**: Linear regression using stochastic gradient descent
- **`least_squares`**: Least squares regression using normal equations
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: using stochastic gradient descent
- **`reg_logistic_regression`**: Regularized logistic regression

#### `run.py`
Script that generates the exact CSV file submitted on Kaggle.

#### `project1.ipynb`
Python notebook used for tests during this project.

## Run
To get the exact results run the `run.py` file.
###### Load data
`y, tX, ids = load_csv_data(data_path)`
###### Data-preprocessing
`y, tX = remove_outliers(y, tX, [features], [thresholds])`
###### Process data and build model
`idx_jet0, y_jet0, tX_jet0, idx_jet1, y_jet1, tX_jet1, idx_jet2, y_jet2, tX_jet2 = separate_jet(y, tX)
tX, rmX = train_data_formatting(tX, degree = 1, cutoff = 0.95, imputation = impute_median, interaction = False)
tX = np.apply_along_axis(standardize, 1, tX)`
###### Apply Polynomial Expansion 
`tX_poly = build_poly(tX, degree)`
###### Ridge regression
`w_ri, mse_ri = ridge_regression(y, tX_poly, lambda_)`
###### Compute label predictions
`y_pred = predict_labels(w_ri, tX_poly)`
###### Merge subsets predictions
`y_pred = merge_jet(idx_jet0, y_pred0, idx_jet1, y_pred1, idx_jet2, y_pred2)`
###### Create submission file
`create_csv_submission(ids_test, y_pred, '../data/final_submission.csv')`


Overleaf link to Report : https://www.overleaf.com/project/5d93529398083800016c9001.
