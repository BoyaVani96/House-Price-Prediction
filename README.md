# House-Price-Prediction
Machine learning project to predict house prices using Random Forest Regression and Streamlit.
# House Price Prediction

A machine learning project that predicts house selling prices using Random Forest Regression and Streamlit.

## Project Overview

This project develops a supervised machine learning regression model to predict the selling price of houses based on different property characteristics.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, data preprocessing, model training, model evaluation, hyperparameter tuning, model saving, and a Streamlit web application.

## Objective

The objective of this project is to build a machine learning model that can estimate the selling price of a house based on its available features such as living area, overall quality, number of bedrooms, bathrooms, garage capacity, neighborhood, and other property characteristics.

## Problem Statement

House prices depend on multiple factors such as property size, quality, location, number of rooms, garage capacity, and age of the house.

The goal of this project is to learn the relationship between these features and historical house prices and use the learned patterns to predict prices for new houses.

This is a supervised learning regression problem.

## Dataset

The project uses the Kaggle House Prices: Advanced Regression Techniques dataset, based on residential homes in Ames, Iowa.

The training dataset contains:

- 1,460 houses
- 81 original columns
- 79 explanatory variables
- `SalePrice` as the target variable

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Git
- GitHub
- VS Code

## Machine Learning Workflow

The project follows this workflow:

1. Problem Definition
2. Dataset Collection
3. Dataset Loading
4. Data Understanding
5. Data Cleaning
6. Exploratory Data Analysis
7. Feature Engineering
8. Data Preprocessing
9. Train/Test Split
10. Model Training
11. Prediction
12. Model Evaluation
13. Hyperparameter Tuning
14. Model Saving
15. Streamlit Application
16. GitHub Version Control

## Data Cleaning

Missing values were handled based on the meaning of each feature.

Categorical features representing the absence of a property component were filled with `"None"`.

Numerical missing values were handled using median values, while the missing categorical value in `Electrical` was handled using the mode.

After cleaning, no missing values remained in the dataset.

## Exploratory Data Analysis

The following analysis was performed:

- Sale price distribution
- Living area vs sale price
- Correlation analysis
- Correlation heatmap
- Overall quality vs sale price
- Living area outlier analysis
- Neighborhood vs sale price analysis

Important features with strong relationships to `SalePrice` included:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `TotRmsAbvGrd`
- `YearBuilt`

## Feature Engineering

Additional features were created to improve the model:

### TotalSF

Combined basement, first-floor, and second-floor areas.

### TotalBathrooms

Combined full and half bathrooms, including basement bathrooms.

### HouseAge

Calculated using:

`Year Sold - Year Built`

### TotalPorchSF

Combined different porch and deck areas.

## Data Preprocessing

The following preprocessing techniques were used:

- Median imputation for numerical features
- Most-frequent imputation for categorical features
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer to combine numerical and categorical preprocessing

## Models

Two main approaches were evaluated:

### Linear Regression

A baseline regression model was trained first.

### Random Forest Regression

Random Forest performed significantly better than the Linear Regression baseline.

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation.

Best parameters:

```text
n_estimators = 100
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
