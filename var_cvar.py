import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the stocks and the timeframe
stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
start_date = '2021-01-01'
end_date = '2024-11-01'

# Fetch historical price data
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio weights (assuming equal allocation)
weights = np.array([1/3, 1/3, 1/3])

# Portfolio returns
portfolio_returns = returns.dot(weights)

# Calculate Value at Risk (VaR) at 95% confidence level
confidence_level = 0.95
VaR = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

# Calculate Conditional Value at Risk (CVaR)
CVaR = portfolio_returns[portfolio_returns <= VaR].mean()

# Output results
print(f"Portfolio VaR at {confidence_level*100}% confidence level: {VaR:.2%}")
print(f"Portfolio CVaR at {confidence_level*100}% confidence level: {CVaR:.2%}")

# Plotting the distribution of returns
plt.figure(figsize=(10,6))
plt.hist(portfolio_returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(VaR, color='red', linestyle='--', label=f'VaR ({VaR:.2%})')
plt.axvline(CVaR, color='orange', linestyle='--', label=f'CVaR ({CVaR:.2%})')
plt.title('Portfolio Returns Distribution with VaR and CVaR')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Function to calculate VaR and CVaR
def calculate_var_cvar(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

# Fetch real-time stock data (e.g., Reliance Industries)
ticker = 'AHL.NS'
start_date = '2021-01-01'
end_date = '2024-11-04'

# Get the stock data
data = yf.download(ticker, start=start_date, end=end_date)
data['Returns'] = data['Adj Close'].pct_change().dropna()

# Calculate VaR and CVaR
var, cvar = calculate_var_cvar(data['Returns'].dropna(), confidence_level=0.95)

# Prepare data for visualization
data['VaR'] = var
data['CVaR'] = cvar

# Create a plot with hover feature
fig = go.Figure()

# Add returns trace
fig.add_trace(go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns',
                         hoverinfo='x+y', line=dict(color='blue')))

# Add VaR trace
fig.add_trace(go.Scatter(x=data.index, y=[var]*len(data), mode='lines', name='Value at Risk',
                         hoverinfo='x+y', line=dict(color='red', dash='dash')))

# Add CVaR trace
fig.add_trace(go.Scatter(x=data.index, y=[cvar]*len(data), mode='lines', name='Conditional VaR',
                         hoverinfo='x+y', line=dict(color='green', dash='dash')))

# Update layout
fig.update_layout(title=f'Value at Risk and Conditional VaR for {ticker}',
                  xaxis_title='Date',
                  yaxis_title='Returns',
                  showlegend=True)

# Show the plot
fig.show()

# Print VaR and CVaR
print(f'Value at Risk (95%): {var:.2%}')
print(f'Conditional Value at Risk (95%): {cvar:.2%}')

