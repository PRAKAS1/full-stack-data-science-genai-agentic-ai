import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    HuberRegressor
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline  # data leakages
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(BASE_DIR / 'USA_Housing.csv')


# Preprocessing
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define a dictionary of models to evaluate
# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    #'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}


# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })

    with (BASE_DIR / f'{name}.pkl').open('wb') as f:
        pickle.dump(model, f)


# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(BASE_DIR / 'model_results.csv', index=False)
print("Model evaluation results saved to 'model_results.csv'.")