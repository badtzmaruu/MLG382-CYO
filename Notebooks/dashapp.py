import os
import pickle
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Initialise
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Weather Rain Prediction"

# Load models and scaler
def load_pickle(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, '..', 'Artifacts', file_name)
    with open(path, 'rb') as f:
        return pickle.load(f)

xgboost_model = load_pickle('xgboost_model.pkl')
random_forest_model = load_pickle('randomforest_model.pkl')
regression_model = load_pickle('regression_model.pkl')
regression_scaler = load_pickle('regression_scaler.pkl')  # Load scaler

#layout
app.layout = dbc.Container([
    html.H1("🌦️ Rain Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Temperature (°C)"),
            dbc.Input(id="temp", type="text", value="20.0"),
            dbc.Label("Humidity (%)"),
            dbc.Input(id="humidity", type="text", value="60.0"),
            dbc.Label("Wind Speed (km/h)"),
            dbc.Input(id="wind", type="text", value="10.0"),
            dbc.Label("Cloud Cover (%)"),
            dbc.Input(id="cloud", type="text", value="50.0"),
            dbc.Label("Pressure (hPa)"),
            dbc.Input(id="pressure", type="text", value="1013.0"),
            dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3")
        ], width=4),

        dbc.Col([
            html.Div(id="prediction-output", className="mt-4")
        ], width=8)
    ])
], fluid=True)

# Callback
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
        return dbc.Alert("❌ Please enter valid numerical values in all fields.", color="danger")

    # Scale input for regression model
    scaled_input = regression_scaler.transform(input_df)

    # Predict with each model
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
