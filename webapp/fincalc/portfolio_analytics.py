import pandas as pd
import numpy as np

def calculate_portfolio_beta(portfolio_holdings, market_data, stock_data):
    """Calculates the market neutrality (Beta) of a portfolio.

    Args:
        portfolio_holdings (pd.DataFrame): DataFrame with columns ['Symbol', 'Value'].
                                           'Value' is the market value of the holding.
        market_data (pd.Series): Series of market index returns. Index must be datetime.
        stock_data (pd.DataFrame): DataFrame of stock returns. Index must be datetime,
                                      columns are stock symbols. Date range should align
                                      with market_data.

    Returns:
        float: The weighted average beta of the portfolio.
        None: If data is insufficient or calculation fails.
    """
    if not isinstance(portfolio_holdings, pd.DataFrame) or \
       not isinstance(market_data, pd.Series) or \
       not isinstance(stock_data, pd.DataFrame):
        print("Error: Invalid input data types.")
        return None

    required_cols = ['Symbol', 'Value']
    if not all(col in portfolio_holdings.columns for col in required_cols):
        print(f"Error: portfolio_holdings DataFrame must contain columns: {required_cols}")
        return None

    if portfolio_holdings.empty or market_data.empty or stock_data.empty:
        print("Error: Input data cannot be empty.")
        return None

    # Ensure alignment - use intersection of indices
    common_index = market_data.index.intersection(stock_data.index)
    if len(common_index) < 2: # Need at least 2 data points for variance/covariance
         print("Error: Insufficient overlapping data points between market and stock returns.")
         return None

    market_returns = market_data.loc[common_index]
    stock_returns = stock_data.loc[common_index]

    # Calculate market variance (denominator for beta)
    market_variance = market_returns.var()
    if market_variance == 0:
        print("Error: Market variance is zero. Cannot calculate Beta.")
        return None

    betas = {}
    for symbol in portfolio_holdings['Symbol'].unique():
        if symbol not in stock_returns.columns:
            print(f"Warning: No return data found for symbol {symbol}. Skipping.")
            betas[symbol] = np.nan # Assign NaN if no data
            continue

        stock_return_series = stock_returns[symbol]

        # Calculate covariance between stock and market returns
        # Ensure series are aligned (should be due to common_index but double-check)
        covariance = stock_return_series.cov(market_returns)

        # Calculate beta for the stock
        beta = covariance / market_variance
        betas[symbol] = beta

    # Map calculated betas back to the holdings DataFrame
    portfolio_holdings['Beta'] = portfolio_holdings['Symbol'].map(betas)

    # Drop holdings where beta couldn't be calculated
    portfolio_holdings = portfolio_holdings.dropna(subset=['Beta'])

    if portfolio_holdings.empty:
        print("Error: Could not calculate Beta for any holdings with available data.")
        return None

    # Calculate portfolio weights
    total_portfolio_value = portfolio_holdings['Value'].sum()
    if total_portfolio_value == 0:
         print("Warning: Total portfolio value is zero. Cannot calculate weighted beta.")
         # Return the average (unweighted) beta in this edge case
         return portfolio_holdings['Beta'].mean() if not portfolio_holdings.empty else 0.0


    portfolio_holdings['Weight'] = portfolio_holdings['Value'] / total_portfolio_value

    # Calculate weighted beta
    portfolio_beta = (portfolio_holdings['Weight'] * portfolio_holdings['Beta']).sum()

    return portfolio_beta
