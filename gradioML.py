import gradio as gr
from lightgbm import LGBMRegressor
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt

# Function to get the payload for the request
def request_payload(method, rows):
    return {
        "username": "Tom",
        "password": "abc123",
        "method": method,
        "rows": rows
    }

# Function to process the request and train models based on the user's choices
def process_request_payload(method, selected_models, rows):
    payload = request_payload(method, int(rows))
    response = requests.post(api_endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        json_data = response.json()
        df = pd.DataFrame(json_data)

        if method == 'getPCBFA':
            # Your logic to preprocess PCBFA data
            df = df.drop(['Open_Circuit', 'DES_Line', 'Scrub_Line', 'Part_No', 'InsertionTime'], axis=1)
            y = df['Open_Circuit']
        elif method == 'getPCBEnig':
            # Your logic to preprocess PCBEnig data
            df = df.drop(['Thickness', 'InsertionTime'], axis=1)
            y = df['Thickness']
        else:
            return "Invalid method"

        X = df  # Assuming all other columns are features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []
        
        if "XGBoost" in selected_models:
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.extend(compute_metrics(y_test, y_pred, 'XGBoost'))

        if "Linear Regression" in selected_models:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.extend(compute_metrics(y_test, y_pred, 'Linear Regression'))

        if "Random Forest" in selected_models:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.extend(compute_metrics(y_test, y_pred, 'Random Forest'))

        if "Decision Tree" in selected_models:
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.extend(compute_metrics(y_test, y_pred, 'Decision Tree'))

        if "LightGBM" in selected_models:
            model = LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.extend(compute_metrics(y_test, y_pred, 'LightGBM'))

        return pd.DataFrame(results, columns=["Model", "Metric", "Value"])
    else:
        return f"Failed to fetch data, status code: {response.status_code}"

# Function to compute metrics
def compute_metrics(y_true, y_pred, model_name):
    return [
        [model_name, "Mean Squared Error (MSE)", mean_squared_error(y_true, y_pred)],
        [model_name, "Root Mean Squared Error (RMSE)", sqrt(mean_squared_error(y_true, y_pred))],
        [model_name, "Mean Absolute Error (MAE)", mean_absolute_error(y_true, y_pred)],
        [model_name, "R^2 Score", r2_score(y_true, y_pred)],
        [model_name, "Median Absolute Error (MedAE)", median_absolute_error(y_true, y_pred)]
    ]

# Gradio app definition
inputs = [
    gr.Dropdown(choices=["getPCBFA", "getPCBEnig"], label="Select Method"),
    gr.CheckboxGroup(choices=["XGBoost", "Linear Regression", "Random Forest", "Decision Tree", "LightGBM"], label="Select Machine Learning Model"),
    gr.Number(label="Number of rows to train the model on")
]

outputs = gr.Dataframe(type="pandas")

# API endpoint and headers
api_endpoint = 'http://192.168.0.232:1880/api/litmus/live'
headers = {'Content-Type': 'application/json'}

gr.Interface(
    fn=process_request_payload,
    inputs=inputs,
    outputs=outputs,
    title="Machine Learning Model Evaluation",
    description="Train and evaluate selected machine learning models on PCB data.",
    allow_flagging=False
).launch()

