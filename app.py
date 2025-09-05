import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go

# Load model
with open('xgb_model_imp.pkl', 'rb') as f:
    model = pickle.load(f)
feature = pd.read_pickle('important_features.pkl')

data = pd.read_csv('data_insolvency.csv')
default_values = data[feature].median().to_list()
min_values = (data[feature].min())
max_values = (data[feature].max())

app = dash.Dash(__name__)
server = app.server

# Generate 14 input fields
input_fields = [
    dcc.Input(
        id=f'input-{i}',
        type='number',
        placeholder=f'{feature[i]}',
        value=round(default_values[i], 2),
        style={'margin': '5px'}
    )
    for i in range(14)
]

app.layout = html.Div([
    html.H2("XGBoost Prediction App"),
    html.Div(input_fields),
    html.Button('Predict', id='predict-btn', style={'margin': '10px'}),
    html.Div(id='output', style={'marginTop': '20px', 'fontWeight': 'bold'}),

    html.Hr(),
    html.H3("🔍 Sensitivity Analysis"),
    html.Label("Select a feature to vary:"),
    dcc.Dropdown(
        id='feature-selector',
        options=[{'label': name, 'value': i} for i, name in enumerate(feature)],
        value=0,
        style={'width': '300px'}
    ),
    # html.Div(id='slider-container'),
    html.Div(
    id='slider-container',
    children=dcc.Slider(
        id='feature-slider',
        min=0,
        max=1,
        step=0.01,
        value=default_values[0],
        marks={int(0): str(0), int(1): str(1)},
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    )
    ),
    dcc.Graph(id='sensitivity-plot')
])

# Prediction callback
@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State(f'input-{i}', 'value') for i in range(14)]
)
def predict(n_clicks, *inputs):
    if n_clicks is None:
        return ""
    # Replace None with default values
    features = [val if val is not None else default_values[i] for i, val in enumerate(inputs)]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return f"✅ Prediction: {prediction[0]}"

# Dynamic slider based on selected feature
@app.callback(
    Output('slider-container', 'children'),
    Input('feature-selector', 'value')
)
def update_slider(selected_feature):
    min_val = min_values[selected_feature]
    max_val = max_values[selected_feature]
    return dcc.Slider(
        id='feature-slider',
        min=min_val,
        max=max_val,
        step=(max_val - min_val) / 100,
        value=default_values[selected_feature],
        marks={int(min_val): str(min_val), int(max_val): str(max_val)},
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    )

# Sensitivity plot callback
@app.callback(
    Output('sensitivity-plot', 'figure'),
    Input('feature-selector', 'value'),
    Input('feature-slider', 'value')
)
def update_sensitivity_plot(selected_feature, slider_value):
    min_val = min_values[selected_feature]
    max_val = max_values[selected_feature]
    values = np.linspace(min_val, max_val, 100)
    predictions = []

    for val in values:
        features = default_values.copy()
        features[selected_feature] = val
        pred = model.predict(np.array(features).reshape(1, -1))[0]
        predictions.append(pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values, y=predictions, mode='lines', name='Prediction'))
    fig.update_layout(
        title=f"Prediction Sensitivity for {feature[selected_feature]}",
        xaxis_title=f"{feature[selected_feature]} Value",
        yaxis_title="Model Prediction",
        template="plotly_white"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
