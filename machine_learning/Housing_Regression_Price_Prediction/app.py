from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', #'LGBM',
    'XGBoost', 'KNN'
]


def load_model(name):
    with (BASE_DIR / f'{name}.pkl').open('rb') as model_file:
        return pickle.load(model_file)


models = {name: load_model(name) for name in model_names}

# Load evaluation results
results_df = pd.read_csv(BASE_DIR / 'model_results.csv')

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])

    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    best_model = results_df.loc[results_df['R2'].idxmax()].to_dict()
    chart_results = []
    for row in results_df.to_dict(orient='records'):
        score = float(row['R2'])
        plotted_score = max(-1.0, min(1.0, score))
        bar_width = abs(plotted_score) * 50
        chart_results.append({
            'model': row['Model'],
            'score': score,
            'score_label': f'{score:.3g}',
            'bar_width': bar_width,
            'bar_left': 50 - bar_width if plotted_score < 0 else 50,
            'is_negative': plotted_score < 0,
            'is_off_scale': score < -1 or score > 1
        })
    chart_results.sort(key=lambda item: item['score'], reverse=True)

    return render_template(
        'model.html',
        results=results_df.to_dict(orient='records'),
        chart_results=chart_results,
        best_model=best_model
    )

if __name__ == '__main__':
    app.run(debug=True)