import pandas as pd

df=pd.read_csv("train.csv")
print(df.head())
print(df.shape)
print(df.columns) 
df.info()

print(df.isnull().sum()[df.isnull().sum() > 0])
none_columns = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature"
]

df[none_columns] = df[none_columns].fillna("None")
print(df.isnull().sum()[df.isnull().sum() > 0])
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

# Fill categorical missing value with the mode
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

# Check remaining missing values
print(df.isnull().sum()[df.isnull().sum() > 0])
print(df["SalePrice"].describe())
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["SalePrice"], kde=True)

plt.title("Distribution of SalePrice")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")

plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])

plt.title("GrLivArea vs SalePrice")
plt.xlabel("Living Area")
plt.ylabel("Sale Price")

plt.show()

correlation = df.select_dtypes(include="number").corr()["SalePrice"]

print(correlation.sort_values(ascending=False))
important_features = [
    "SalePrice",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt"
]

# Calculate correlation
corr_matrix = df[important_features].corr()

# Create heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(x=df["OverallQual"], y=df["SalePrice"])

plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.show()

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["GrLivArea"])

plt.title("GrLivArea Outliers")
plt.xlabel("Living Area")

plt.show()

large_houses = df[df["GrLivArea"] > 4000]

print(large_houses[["GrLivArea", "SalePrice"]])
plt.figure(figsize=(14, 7))

sns.boxplot(x="Neighborhood", y="SalePrice", data=df)

plt.title("Neighborhood vs Sale Price")
plt.xlabel("Neighborhood")
plt.ylabel("Sale Price")

plt.xticks(rotation=45)

plt.show()

numerical_features = df.select_dtypes(include="number").columns
categorical_features = df.select_dtypes(exclude="number").columns

print("Number of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)

print(df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "TotalSF"]].head())

df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

print(df[
    ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath", "TotalBathrooms"]
].head())
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

print(df[["YearBuilt", "YrSold", "HouseAge"]].head())

df["TotalPorchSF"] = (
    df["WoodDeckSF"]
    + df["OpenPorchSF"]
    + df["EnclosedPorch"]
    + df["3SsnPorch"]
    + df["ScreenPorch"]
)

print(df[
    [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "TotalPorchSF"
    ]
].head())
print(
    df[
        ["TotalSF", "TotalBathrooms", "HouseAge", "TotalPorchSF", "SalePrice"]
    ].corr()["SalePrice"].sort_values(ascending=False)
)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("X shape:", X.shape)
print("y shape:", y.shape)

numerical_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(exclude="number").columns

print("Number of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
X_train = X_train.drop("Id", axis=1)
X_test = X_test.drop("Id", axis=1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features.drop("Id")),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
preprocessor.fit(X_train)

print("Preprocessor fitted successfully.")
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_test shape:", X_test_processed.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_processed, y_train)

predictions = model.predict(X_test_processed)

print("First 10 predictions:")
print(predictions[:10])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

rf_predictions = rf_model.predict(X_test_processed)

print("First 10 Random Forest predictions:")
print(rf_predictions[:10])
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = rf_mse ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)
rf_train_predictions = rf_model.predict(X_train_processed)

train_r2 = r2_score(y_train, rf_train_predictions)
test_r2 = r2_score(y_test, rf_predictions)

print("Training R²:", train_r2)
print("Testing R²:", test_r2)
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train_processed, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation R²:")
print(grid_search.best_score_)

best_rf_model = grid_search.best_estimator_

# Predict on the untouched test data
tuned_predictions = best_rf_model.predict(X_test_processed)

# Evaluate the tuned model
tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_mse = mean_squared_error(y_test, tuned_predictions)
tuned_rmse = tuned_mse ** 0.5
tuned_r2 = r2_score(y_test, tuned_predictions)

print("\nTuned Random Forest Results:")
print("MAE:", tuned_mae)
print("MSE:", tuned_mse)
print("RMSE:", tuned_rmse)
print("R² Score:", tuned_r2)

print("\n===== TUNED MODEL TEST =====")

best_rf_model = grid_search.best_estimator_

tuned_predictions = best_rf_model.predict(X_test_processed)

tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_mse = mean_squared_error(y_test, tuned_predictions)
tuned_rmse = tuned_mse ** 0.5
tuned_r2 = r2_score(y_test, tuned_predictions)

print("Tuned Random Forest Results:")
print("MAE:", tuned_mae)
print("MSE:", tuned_mse)
print("RMSE:", tuned_rmse)
print("R² Score:", tuned_r2)

import joblib

# Save the trained model
joblib.dump(best_rf_model, "house_price_model.pkl")

# Save the preprocessing pipeline
joblib.dump(preprocessor, "preprocessor.pkl")

print("Model and preprocessor saved successfully.")
app_features = [
    "GrLivArea",
    "OverallQual",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "TotRmsAbvGrd",
    "YearRemodAdd",
    "Neighborhood",
    "YrSold"
]

X_app = df[app_features]
y_app = df["SalePrice"]

print("Number of app features:", len(app_features))
print("X_app shape:", X_app.shape)
print("y_app shape:", y_app.shape)
from sklearn.model_selection import train_test_split

X_app_train, X_app_test, y_app_train, y_app_test = train_test_split(
    X_app,
    y_app,
    test_size=0.2,
    random_state=42
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

app_numerical_features = [
    "GrLivArea",
    "OverallQual",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "TotRmsAbvGrd",
    "YearRemodAdd",
    "YrSold"
]

app_categorical_features = [
    "Neighborhood"
]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

app_numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

app_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

app_preprocessor = ColumnTransformer(
    transformers=[
        ("num", app_numerical_pipeline, app_numerical_features),
        ("cat", app_categorical_pipeline, app_categorical_features)
    ]
)

app_preprocessor.fit(X_app_train)

print("App preprocessor fitted successfully.")
X_app_train_processed = app_preprocessor.transform(X_app_train)
X_app_test_processed = app_preprocessor.transform(X_app_test)

print("Processed app training shape:", X_app_train_processed.shape)
print("Processed app testing shape:", X_app_test_processed.shape)
from sklearn.ensemble import RandomForestRegressor

app_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

app_model.fit(X_app_train_processed, y_app_train)

print("App Random Forest model trained successfully.")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app_predictions = app_model.predict(X_app_test_processed)

app_mae = mean_absolute_error(y_app_test, app_predictions)
app_mse = mean_squared_error(y_app_test, app_predictions)
app_rmse = app_mse ** 0.5
app_r2 = r2_score(y_app_test, app_predictions)

print("\n===== APP MODEL RESULTS =====")
print("MAE:", app_mae)
print("MSE:", app_mse)
print("RMSE:", app_rmse)
print("R² Score:", app_r2)
import joblib

# Save the app model
joblib.dump(app_model, "house_price_app_model.pkl")

# Save the app preprocessor
joblib.dump(app_preprocessor, "app_preprocessor.pkl")

print("App model and preprocessor saved successfully.")