import os
import pickle
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Weather Rain Prediction"

#loading models and scalers.
def load_pickle(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, '..', 'Artifacts', file_name)
    with open(path, 'rb') as f:
        return pickle.load(f)
xgboost_model = load_pickle('xgboost_model.pkl')
random_forest_model = load_pickle('randomforest_model.pkl')
regression_model = load_pickle('regression_model.pkl')
regression_scaler = load_pickle('regression_scaler.pkl')

#layour of dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🌦️ Rain Prediction", className="text-center mb-4"),
            dbc.Label("Temperature (°C)"),
            dbc.Input(id="temp", type="number", value="20.0", step="0.1"),
            dbc.Label("Humidity (%)"),
            dbc.Input(id="humidity", type="number", value="60.0", step="0.1"),
            dbc.Label("Wind Speed (km/h)"),
            dbc.Input(id="wind", type="number", value="10.0", step="0.1"),
            dbc.Label("Cloud Cover (%)"),
            dbc.Input(id="cloud", type="number", value="50.0", step="0.1"),
            dbc.Label("Pressure (hPa)"),
            dbc.Input(id="pressure", type="number", value="1013.0", step="0.1"),
            dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3"),
            html.Div(id="input-error", className="mt-3"),
            html.Div(id="prediction-output", className="mt-4 text-center")
        ], width=6)
    ], justify="center", className="d-flex align-items-center min-vh-100")
], fluid=True)
#validation'
@app.callback(
    Output("input-error", "children"),
    Input("temp", "value"),
    Input("humidity", "value"),
    Input("wind", "value"),
    Input("cloud", "value"),
    Input("pressure", "value")
)
def validate_inputs(temp, humidity, wind, cloud, pressure):
    fields = {
        "Temperature": temp,
        "Humidity": humidity,
        "Wind Speed": wind,
        "Cloud Cover": cloud,
        "Pressure": pressure
    }
    try:
        temp = float(temp)
        humidity = float(humidity)
        wind = float(wind)
        cloud = float(cloud)
        pressure = float(pressure)
    except (ValueError, TypeError):
        return dbc.Alert("Please enter valid numbers in all fields.", color="danger")
    if any(val is None or str(val).strip() == "" for val in fields.values()):
        return dbc.Alert("All fields are required.", color="danger")
    # Range validations
    if not (-50 <= temp <= 60):
        return dbc.Alert("Temperature must be between -50 and 60 C.", color="danger")
    if not (0 <= humidity <= 100):
        return dbc.Alert("Humidity must be between 0 and 100%.", color="danger")
    if not (0 <= wind <= 200):
        return dbc.Alert("Wind Speed must be between 0 and 200 km per hour.", color="danger")
    if not (0 <= cloud <= 100):
        return dbc.Alert("Cloud Cover must be between 0 and 100%.", color="danger")
    if not (800 <= pressure <= 1100):
        return dbc.Alert("Pressure must be between 800 and 1100 hPa.", color="danger")

    return None
# calback
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("temp", "value"),
    State("humidity", "value"),
    State("wind", "value"),
    State("cloud", "value"),
    State("pressure", "value"),
    prevent_initial_call=True
)
def make_prediction(n_clicks, temp, humidity, wind, cloud, pressure):
    try:
        input_df = pd.DataFrame([{
            "Temperature": float(temp),
            "Humidity": float(humidity),
            "Wind_Speed": float(wind),
            "Cloud_Cover": float(cloud),
            "Pressure": float(pressure)
        }])
    except (ValueError, TypeError):
        return dbc.Alert("Please enter valid numerical values in all fields.", color="danger")
    #input to use scaler for th regression model
    scaled_input = regression_scaler.transform(input_df)

    # prediction for each model
    preds = {
        "XGBoost": xgboost_model.predict(input_df)[0],
        "Random Forest": random_forest_model.predict(input_df)[0],
        "Regression": regression_model.predict(scaled_input)[0]
    }
    readable_preds = {
        model: "🌧️ Rain" if str(pred) in ["1", "Rain"] else "☀️ No Rain"
        for model, pred in preds.items()
    }
    return html.Div([
        html.H4("Predictions"),
        html.Ul([
            html.Li(f"{model}: {result}")
            for model, result in readable_preds.items()
        ])
    ])
if __name__ == "__main__":
    app.run(debug=True)
