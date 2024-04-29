# Financial-Data-Analytics-Personal-Project
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from scipy.optimize import minimize
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import datetime

# Set the current 10-year Treasury rate as the risk-free rate
risk_free_rate = 0.0461  # As of April 23, 2024

# Define risk level constraints with dynamic allocation based on asset volatility
risk_level_constraints = {
    'low': 0.15,
    'medium': 0.25,
    'high': 0.35
}

# External stylesheets for better styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize Dash app with external stylesheets for better styling
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Enhanced function to fetch stock data with caching
@lru_cache(maxsize=100)
def fetch_data(symbols, start_date, end_date):
    """Fetch stock data using yfinance with caching and improved error handling."""
    try:
        symbol_string = ','.join(sorted(set(symbols))) if isinstance(symbols, list) else symbols
        data = yf.download(symbol_string, start=start_date, end=end_date)
        if data.empty:
            return pd.DataFrame(), "No data fetched; check the symbols or date range."
        # Debugging: Print data columns to check structure
        print(data.columns)
        return data, None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {str(e)}"


# Calculate returns from data
def calculate_returns(data):
    """Calculate and return the percentage returns from adjusted close prices."""
    if data.empty:
        return pd.DataFrame()

    # Check if the DataFrame has a MultiIndex on columns
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker scenario: use cross-section on 'Adj Close'
        returns = data.xs('Adj Close', level=0, axis=1).pct_change().dropna()
    else:
        # Single-ticker scenario: directly access 'Adj Close'
        if 'Adj Close' in data.columns:
            returns = data['Adj Close'].pct_change().dropna()
        else:
            # No 'Adj Close' column found, return empty DataFrame
            return pd.DataFrame()

    return returns


# Updated function for diversification analysis
def diversification_analysis(returns, weights):
    """Performs diversification analysis on the given returns using the specified weights."""
    try:
        cov_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights.reshape(-1, 1))))
        individual_volatilities = np.sqrt(np.diag(cov_matrix))
        weighted_sum_volatilities = np.sum(weights * individual_volatilities)
        diversification_ratio = portfolio_volatility / weighted_sum_volatilities
        hhi = np.sum(weights**2)
        effective_number = 1 / hhi
        correlation_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis'
        ))
        fig.update_layout(title="Correlation Matrix of Asset Returns",
                          xaxis_title="Assets",
                          yaxis_title="Assets")

        analysis_results = {
            'portfolio_volatility': portfolio_volatility,
            'diversification_ratio': diversification_ratio,
            'hhi': hhi,
            'effective_number': effective_number,
            'correlation_matrix_fig': fig
        }
        return analysis_results, None
    except Exception as e:
        return {}, f"Error in diversification analysis: {str(e)}"

# Function to optimize portfolio using Sharpe ratio
def optimize_portfolio(returns, risk_free_rate, risk_level='medium'):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    print("Cov_matrix null check:", cov_matrix.isnull().any().any())  # Ensure this prints a single boolean

    if mean_returns.isnull().any() or cov_matrix.isnull().any().any():
        return None, "Invalid return data or insufficient data for covariance calculation."

    # [rest of your code]

    print("Type of mean_returns:", type(mean_returns))
    print("Type of cov_matrix:", type(cov_matrix))
    print("Is null in mean_returns:", mean_returns.isnull().any())
    print("Is null in cov_matrix:", cov_matrix.isnull().any())

    # Check if there are any null values in mean_returns or cov_matrix
    if mean_returns.isnull().any() or cov_matrix.isnull().any().any():
        return None, "Invalid return data or insufficient data for covariance calculation."

    def negative_sharpe_ratio(weights):
        p_ret = np.dot(weights, mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if p_vol == 0:
            return float('inf')  # Avoid division by zero by returning infinity
        return -(p_ret - risk_free_rate) / p_vol

    num_assets = len(mean_returns)
    max_weight = risk_level_constraints.get(risk_level, 0.25)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: max_weight - max(x)}
    ]
    bounds = [(0, 1) for _ in range(num_assets)]  # Allowing weights to fully range from 0 to 1
    initial_guess = np.full(num_assets, 1 / num_assets)
    options = {'disp': True, 'maxiter': 1000}  # Display output and increase max iterations

    result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
    if result.success:
        return result.x, None
    else:
        return None, f"Optimization failed: {result.message}"


# Define app layout
app.layout = html.Div([
    html.H1('Portfolio Optimization Dashboard', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label('Enter Tickers (comma-separated):'),
            dcc.Input(id='tickers-input', type='text', value='AAPL, MSFT', style={'width': '100%'}),
        ], style={'padding': '10px'}),
        
        html.Div([
            html.Label('Select Risk Level:'),
            dcc.Dropdown(
                id='risk-level-dropdown',
                options=[
                    {'label': 'Low', 'value': 'low'},
                    {'label': 'Medium', 'value': 'medium'},
                    {'label': 'High', 'value': 'high'}
                ],
                value='medium'
            ),
        ], style={'padding': '10px'}),
        
        html.Div([
            html.Label('Start Date:'),
            dcc.DatePickerSingle(
                id='start-date-input',
                date=datetime.date(2023, 1, 1),
            ),
        ], style={'padding': '10px'}),
        
        html.Div([
            html.Label('End Date:'),
            dcc.DatePickerSingle(
                id='end-date-input',
                date=datetime.date(2024, 1, 1),
            ),
        ], style={'padding': '10px'}),
        
        html.Button('Optimize Portfolio', id='optimize-button', n_clicks=0, style={'margin': '20px'}),
    ], style={'display': 'flex', 'flex-direction': 'column', 'max-width': '500px', 'margin': 'auto'}),
    
    html.Div(id='portfolio-performance-chart', style={'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
    html.Div(id='risk-assessment', style={'padding': '20px'}),
    html.Div(id='diversification-analysis', style={'padding': '20px'}),
    html.Div(id='evaluation-results', style={'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif'})

# Define callbacks to dynamically update components
@app.callback(
    [
        Output('portfolio-performance-chart', 'children'),
        Output('risk-assessment', 'children'),
        Output('diversification-analysis', 'children'),
        Output('evaluation-results', 'children')
    ],
    [
        Input('optimize-button', 'n_clicks')
    ],
    [
        State('tickers-input', 'value'),
        State('risk-level-dropdown', 'value'),
        State('start-date-input', 'date'),
        State('end-date-input', 'date')
    ]
)
def optimize_portfolio_callback(n_clicks, tickers_input, risk_level, start_date, end_date):
    if n_clicks == 0:
        raise PreventUpdate

    tickers = tuple(ticker.strip() for ticker in tickers_input.split(','))
    data, error = fetch_data(tickers, start_date, end_date)
    
    if error:
        return html.Div(f"Error: {error}"), None, None, None

    if data.empty:
        return html.Div("No data available for the selected date range or tickers."), None, None, None

    try:
        returns = calculate_returns(data)
        if returns.empty:
            return html.Div("No returns data available."), None, None, None
    except Exception as e:
        return html.Div(f"Unexpected error while calculating returns: {str(e)}"), None, None, None

    optimized_weights, opt_error = optimize_portfolio(returns, risk_free_rate, risk_level)
    if opt_error:
        return html.Div(f"Optimization Error: {opt_error}"), None, None, None

    analysis_results, analysis_error = diversification_analysis(returns, optimized_weights)
    if analysis_error:
        return html.Div(f"Analysis Error: {analysis_error}"), None, None, None

    portfolio_performance_chart = dcc.Graph(
        figure=go.Figure(data=[go.Scatter(x=returns.index, y=returns.cumsum(), mode='lines')]),
        style={'width': '100%', 'height': '400px'}
    )

    risk_assessment = html.Div([
        html.H3('Risk Assessment'),
        html.P(f"Portfolio Volatility: {analysis_results['portfolio_volatility']:.2f}")
    ])

    diversification_contents = html.Div([
        html.H3("Diversification Analysis"),
        dcc.Graph(figure=analysis_results['correlation_matrix_fig'])
    ])

    evaluation_results = html.Div([
        html.H3('Evaluation Results'),
        html.P(f"Diversification Ratio: {analysis_results['diversification_ratio']:.2f}"),
        html.P(f"HHI: {analysis_results['hhi']:.2f}")
    ])

    return portfolio_performance_chart, risk_assessment, diversification_contents, evaluation_results


if __name__ == '__main__':
    app.run_server(debug=True)
