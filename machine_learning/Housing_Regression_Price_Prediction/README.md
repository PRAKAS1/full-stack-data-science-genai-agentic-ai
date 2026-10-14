# Homewise: Housing Price Prediction

Homewise is a Flask application that demonstrates housing-price regression. It uses the included `USA_Housing.csv` dataset to train several regression models, compare their metrics, and generate an estimate from five numeric inputs.

> **Educational project only:** estimates use area-level averages and are not appraisals, offers, or financial advice. Do not use them for buying, selling, lending, insurance, or investment decisions.

## Project contents

- `app.py` — Flask application and prediction/model-results routes.
- `model.py` — trains models, evaluates them on a held-out test split, and saves model files plus `model_results.csv`.
- `USA_Housing.csv` — dataset used for training.
- `templates/` — estimator, prediction, model scorecard, and project information pages.
- `static/css/style.css` — responsive application styles.
- `*.pkl` — serialized trained models loaded by the app.
- `model_results.csv` — saved evaluation metrics shown on the scorecard.

## Requirements

- Python 3.10 or newer
- Flask
- pandas
- scikit-learn
- LightGBM
- XGBoost

Install the Python dependencies in a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install Flask pandas scikit-learn lightgbm xgboost
```

If PowerShell prevents activating the virtual environment, run its interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install Flask pandas scikit-learn lightgbm xgboost
```

## Run the application

From this project directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000/` in a browser. The application also provides:

- `/results` — model evaluation chart and metrics table.
- `/info` — project purpose, input features, metrics, and limitations.

The app expects all model `.pkl` files and `model_results.csv` to be present in this directory. These files are included with the project. If they are missing or you want to retrain:

```powershell
.\.venv\Scripts\python.exe model.py
```

Training can take a while because multiple models are fitted. The script saves the trained models and evaluation CSV alongside itself.

## Inputs and evaluation

The prediction form accepts average area income, average house age, average number of rooms, average number of bedrooms, and area population. `model.py` uses `Price` as the target and drops `Address` from the feature data. It creates an 80/20 train/test split with `random_state=0` and reports:

- **MAE (mean absolute error):** average absolute prediction error; lower is better.
- **MSE (mean squared error):** average squared prediction error; lower is better.
- **R²:** explained variation score; higher is better. Negative values indicate performance worse than predicting the test-set mean.

The scorecard reflects one split of this dataset, not a guarantee of performance on other areas or future home prices.

## Mermaid workflow diagram

The home page renders its workflow diagram using Mermaid from jsDelivr, so an internet connection is required for that diagram. The rest of the application does not rely on Mermaid to serve predictions or model results.
