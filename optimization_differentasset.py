

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize

# Define Indian assets and ETFs (use tickers available in NSE)
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS','ITC.NS', 'HCC.NS']  # Example: stocks and gold ETF
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio optimization function
def portfolio_optimization(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized
    return portfolio_volatility

# Constraints: weights sum to 1, all weights between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(tickers)))
initial_weights = [1/len(tickers)] * len(tickers)

# Run optimization
optimized = minimize(portfolio_optimization, initial_weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Calculate optimal portfolio performance
optimal_return = np.sum(returns.mean() * optimal_weights) * 252
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))

print(f"Optimal Weights: {optimal_weights}")
print(f"Expected Annual Return: {optimal_return:.2%}")
print(f"Expected Annual Volatility: {optimal_volatility:.2%}")

# Efficient Frontier with Plotly for interactive hover
def efficient_frontier(returns, num_portfolios=5000):
    results = np.zeros((3, num_portfolios))
    all_weights = []
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - 0.01) / portfolio_volatility  # Sharpe ratio
        all_weights.append(weights)
    return results, all_weights

# Generate efficient frontier results
results, all_weights = efficient_frontier(returns)

# Plot efficient frontier with Plotly
fig = go.Figure()

# Scatter plot of the efficient frontier
fig.add_trace(go.Scatter(
    x=results[0, :],
    y=results[1, :],
    mode='markers',
    marker=dict(
        color=results[2, :],  # Color by Sharpe Ratio
        colorscale='Viridis',
        colorbar=dict(title='Sharpe Ratio'),
        showscale=True
    ),
    text=[f"Weights: {np.round(w, 2)}" for w in all_weights],  # Hover text with weights
    hovertemplate='<b>Volatility:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<br>%{text}'
))

# Highlight the optimal portfolio
fig.add_trace(go.Scatter(
    x=[optimal_volatility],
    y=[optimal_return],
    mode='markers',
    marker=dict(color='red', size=15, symbol='star'),
    name='Optimal Portfolio',
    hovertemplate='<b>Optimal Portfolio</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Weights: ' + str(np.round(optimal_weights, 2))
))

fig.update_layout(
    title='Efficient Frontier with Multiple Asset Classes',
    xaxis_title='Volatility',
    yaxis_title='Return'
)

fig.show()

!pip install plotly

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objs as go

# Set up ticker symbols for Indian stocks and other asset classes (e.g., gold ETFs)
tickers = ['ITC.NS', 'ASIANPAINT.NS', 'HDFCBANK.NS', 'SILVERBEES.NS']  # Example tickers for stocks and gold ETF
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio optimization function (minimize volatility)
def portfolio_optimization(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized
    return portfolio_volatility

# Constraints: weights sum to 1, all weights between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(tickers)))
initial_weights = [1/len(tickers)] * len(tickers)

# Run optimization
optimized = minimize(portfolio_optimization, initial_weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Calculate optimal portfolio performance
optimal_return = np.sum(returns.mean() * optimal_weights) * 252
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))

# Display results
print(f"Optimal Weights: {optimal_weights}")
print(f"Expected Annual Return: {optimal_return:.2%}")
print(f"Expected Annual Volatility: {optimal_volatility:.2%}")

# Efficient Frontier with Interactivity using Plotly
def efficient_frontier(returns, num_portfolios=5000):
    results = {'volatility': [], 'return': [], 'sharpe': [], 'weights': []}
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
        results['volatility'].append(portfolio_volatility)
        results['return'].append(portfolio_return)
        results['sharpe'].append(sharpe_ratio)
        results['weights'].append(weights)
    return results

# Get efficient frontier results
results = efficient_frontier(returns)

# Plotting with Plotly
fig = go.Figure()

# Scatter plot for portfolios on efficient frontier
fig.add_trace(go.Scatter(
    x=results['volatility'], y=results['return'],
    mode='markers',
    marker=dict(color=results['sharpe'], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
    text=[f"Weights: {np.round(w, 2)}" for w in results['weights']],
    hoverinfo='text'
))

# Highlight the optimal portfolio
fig.add_trace(go.Scatter(
    x=[optimal_volatility], y=[optimal_return],
    mode='markers',
    marker=dict(color='red', size=12, symbol='star'),
    name='Optimal Portfolio',
    text=[f"Weights: {np.round(optimal_weights, 2)}"],
    hoverinfo='text'
))

# Layout settings for the plot
fig.update_layout(
    title='Efficient Frontier with Optimal Portfolio',
    xaxis=dict(title='Annualized Volatility'),
    yaxis=dict(title='Annualized Return'),
    showlegend=True
)

fig.show()

