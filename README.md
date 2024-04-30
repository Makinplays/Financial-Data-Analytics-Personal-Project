import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import datetime
import matplotlib.pyplot as plt
import base64  # Import base64 module for encoding

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

# Initialize Dash app with external stylesheets
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
        return data, None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {str(e)}"

# Calculate returns from data
def calculate_returns(data):
    """Calculate and return the percentage returns from adjusted close prices."""
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        returns = data.xs('Adj Close', level=0, axis=1).pct_change().dropna()
    else:
        returns = data['Adj Close'].pct_change().dropna() if 'Adj Close' in data.columns else pd.DataFrame()
    return returns

# Updated function for diversification analysis
# Performs diversification analysis on the given returns using the specified weights.
def diversification_analysis(returns, weights):
    try:
        cov_matrix = returns.cov()
        # Ensure weights is a proper 1D array
        weights = np.array(weights).flatten()
        # Calculate portfolio volatility as a scalar
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
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
            'portfolio_volatility': portfolio_volatility.item(),  # Ensures scalar output
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

    # Define the negative Sharpe ratio function
    def negative_sharpe_ratio(weights):
        p_ret = np.dot(weights, mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - risk_free_rate) / p_vol if p_vol > 0 else float('inf')

    num_assets = len(mean_returns)
    bounds = Bounds([0] * num_assets, [1] * num_assets)
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'ineq', 'fun': lambda weights: risk_level_constraints[risk_level] - max(weights)}
    ]
    initial_guess = np.full(num_assets, 1 / num_assets)
    options = {'maxiter': 5000, 'disp': True}

    result = minimize(negative_sharpe_ratio, initial_guess, method='trust-constr',
                      bounds=bounds, constraints=constraints, options=options)

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
    print("Data:", data)
    print("Error:", error)
    
    if error:
        return html.Div(f"Error: {error}"), None, None, None

    if data.empty:
        return html.Div("No data available for the selected date range or tickers."), None, None, None

    try:
        returns = calculate_returns(data)
        print("Returns:", returns)
        if returns.empty:
            return html.Div("No returns data available."), None, None, None
    except Exception as e:
        return html.Div(f"Unexpected error while calculating returns: {str(e)}"), None, None, None

    optimized_weights, opt_error = optimize_portfolio(returns, risk_free_rate, risk_level)
    print("Optimized Weights:", optimized_weights)
    print("Optimization Error:", opt_error)
    if opt_error:
        return html.Div(f"Optimization Error: {opt_error}"), None, None, None

    analysis_results, analysis_error = diversification_analysis(returns, optimized_weights)
    print("Analysis Results:", analysis_results)
    print("Analysis Error:", analysis_error)
    if analysis_error:
        return html.Div(f"Analysis Error: {analysis_error}"), None, None, None

    # Updated portfolio_performance_chart with better date formatting and y-axis as percentage
    # Plot cumulative returns using Matplotlib
    plt.figure(figsize=(14, 7))
    for c in returns.columns.values:
        plt.plot(returns.index, (1 + returns[c]).cumprod() - 1, lw=3, alpha=0.8, label=c)
    plt.title('Cumulative Returns Over Time')
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Date')
    plt.gcf().autofmt_xdate()  # Rotate dates for better readability

    # Convert the Matplotlib plot to base64
    plt_base64 = plt_to_base64(plt)
    plt.close()  # Close the plot to prevent it from displaying in the Dash app

    # Convert the base64 plot to HTML
    portfolio_performance_chart = html.Div([html.Img(src="data:image/png;base64,{}".format(plt_base64))],
                                           style={'width': '100%', 'text-align': 'center'})

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

def plt_to_base64(plt):
    """Convert a Matplotlib plot to base64 encoded image."""
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return plot_base64

if __name__ == '__main__':
    app.run_server(debug=True)
