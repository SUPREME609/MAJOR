import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np

# Load dataset and train model
df = pd.read_csv("Project_data.csv")
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('columntransformer', ct),
    ('RandomForestRegressor', RandomForestRegressor())
])

pipeline.fit(X, y)

# Save model
pickle.dump(pipeline, open('RFR.pkl', 'wb'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model and data
model = pickle.load(open('RFR.pkl', 'rb'))
car = pd.read_csv('Project_data.csv')

@app.route('/')
def home():
    return render_template('main file.html')

@app.route('/second')
def second():
    return render_template('login.html')

@app.route('/buy')
def buy():
    return render_template('buy.html')

@app.route('/sale')
def sale():
    return render_template('salerefur.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/parts')
def parts():
    return render_template('parts.html')

@app.route('/sell')
def sell():
    return render_template('sell.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/hyundai')
def hyundai():
    return render_template('hyundaicreta.html')

@app.route('/hondacity')
def hondacity():
    return render_template('hondacity.html')

@app.route('/toyotacamry')
def toyotacamry():
    return render_template('toyotacamry.html')

@app.route('/first')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Build mapping from company to car models
    car_models_by_company = {}
    for company in companies:
        models = sorted(car[car['company'] == company]['name'].unique())
        car_models_by_company[company] = models

    return render_template('index.html',
                           companies=companies,
                           years=years,
                           fuel_types=fuel_types,
                           car_models_by_company=car_models_by_company)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form['company']
        car_model = request.form['car_models']
        year = int(request.form['year'])
        fuel_type = request.form['fuel_type']
        driven = int(request.form['kilo_driven'])

        input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(input_df)
        return str(np.round(prediction[0], 2))

    except Exception as e:
        print("🔥 Prediction Error:", e)
        return f"₹Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
