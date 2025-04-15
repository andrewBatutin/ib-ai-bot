import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf

# Configure logging for the module if not already configured
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

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


def dump_portfolio_to_json(base_api_url: str, account_id: str, output_dir_relative: str = "data") -> bool:
    """Fetches portfolio positions, converts to a pandas DataFrame, and saves as JSON.

    Args:
        base_api_url (str): The base URL for the IBKR API.
        account_id (str): The account ID to fetch positions for.
        output_dir_relative (str): The directory path relative to the project root
                                   where the JSON file will be saved. Defaults to "data".

    Returns:
        bool: True if the portfolio was successfully dumped, False otherwise.
    """
    logger.info(f"Attempting to dump portfolio for account {account_id}")

    try:
        # Construct the full URL
        positions_url = f"{base_api_url}/portfolio/{account_id}/positions/0"
        logger.debug(f"Fetching positions from: {positions_url}")

        # Make the API request (disable SSL verification as in app.py)
        response = requests.get(positions_url, verify=False)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        positions_raw = response.json()

        if not positions_raw or not isinstance(positions_raw, list):
            logger.warning(f"No positions data received or data is not a list for account {account_id}.")
            return False

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(positions_raw)
        logger.info(f"Successfully fetched {len(portfolio_df)} positions.")

        # Define the output path
        # Assume this script is run from project root or path is relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        output_path = os.path.join(project_root, output_dir_relative)

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_path}")

        # Define the filename
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"portfolio_{current_date}.json"
        full_filepath = os.path.join(output_path, filename)

        # Save the DataFrame to JSON
        # Use orient='records' for a list of dicts format, which is common
        portfolio_df.to_json(full_filepath, orient='records', indent=4)
        logger.info(f"Portfolio successfully saved to: {full_filepath}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"API Error fetching positions for account {account_id}: {e}")
        return False
    except ValueError as e: # Handles JSON decoding errors
        logger.error(f"JSON Decode Error fetching positions: {e}")
        return False
    except OSError as e:
        logger.error(f"File System Error saving portfolio to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during portfolio dump: {e}", exc_info=True)
        return False

def calculate_dollar_neutrality(portfolio_df: pd.DataFrame) -> Optional[float]:
    """
    Calculates the approximate dollar neutrality of a portfolio from a DataFrame.

    Sums the 'mktValue' for all positions in the DataFrame.
    A value close to zero indicates approximate dollar neutrality.

    Args:
        portfolio_df: pandas DataFrame containing portfolio positions.
                      Must contain a column named 'mktValue'.

    Returns:
        The total market value (sum of 'mktValue') as a float,
        or None if the input is invalid or the 'mktValue' column is missing/empty.
    """
    if not isinstance(portfolio_df, pd.DataFrame):
        logger.error(f"Error: Input must be a pandas DataFrame, got {type(portfolio_df)}")
        return None

    if portfolio_df.empty:
        logger.warning("Warning: Input DataFrame is empty.")
        return 0.0 # Return 0 for an empty portfolio

    if 'mktValue' not in portfolio_df.columns:
        logger.error("Error: DataFrame must contain a 'mktValue' column.")
        return None

    # Attempt to convert mktValue column to numeric, coercing errors to NaN
    # This handles cases where the column might be object type with non-numeric strings
    mkt_values_numeric = pd.to_numeric(portfolio_df['mktValue'], errors='coerce')

    # Check if all values became NaN after coercion (meaning no valid numeric data)
    if mkt_values_numeric.isnull().all():
        logger.error("Error: 'mktValue' column contains no valid numeric data.")
        return None

    # Calculate the sum, skipping any NaN values that might have resulted from coercion
    total_value = mkt_values_numeric.sum(skipna=True)

    logger.info(f"Calculated total market value (dollar neutrality): {total_value}")
    return float(total_value)


def calculate_stock_volatility(tickers: List[str], period: str = "1y") -> Dict[str, Optional[float]]:
    """
    Calculates the daily standard deviation (volatility) of stock returns.

    Args:
        tickers (List[str]): A list of stock ticker symbols.
        period (str): The period over which to fetch data (e.g., "1mo", "6mo", "1y", "2y", "max").
                      Defaults to "1y".

    Returns:
        Dict[str, Optional[float]]: A dictionary mapping each ticker to its daily standard deviation.
                                     Returns None for a ticker if data is insufficient or invalid.
    """
    volatility_results = {}
    logger.info(f"Calculating volatility for tickers: {tickers} over period: {period}")

    if not tickers:
        logger.warning("No tickers provided.")
        return {}

    try:
        # Download historical data for all tickers at once
        data = yf.download(tickers, period=period, progress=False)

        if data.empty:
            logger.error("Failed to download any data.")
            for ticker in tickers:
                volatility_results[ticker] = None
            return volatility_results

        # Use Adjusted Close prices
        adj_close = data['Close']

        # Handle cases where only one ticker is requested (returns a Series)
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=tickers[0]) # Convert Series to DataFrame

        # Calculate daily percentage returns
        daily_returns = adj_close.pct_change().dropna() # Drop the first NaN row

        if daily_returns.empty:
             logger.warning(f"Not enough data to calculate returns for period {period}.")
             for ticker in tickers:
                 volatility_results[ticker] = None
             return volatility_results

        # Calculate standard deviation for each ticker
        daily_std_dev = daily_returns.std()

        # Populate results dictionary
        for ticker in tickers:
            if ticker in daily_std_dev.index:
                volatility_results[ticker] = daily_std_dev[ticker]
            else:
                # This might happen if yfinance couldn't fetch data for a specific ticker
                logger.warning(f"Could not calculate volatility for {ticker}, data might be missing.")
                volatility_results[ticker] = None

    except Exception as e:
        logger.error(f"An error occurred during volatility calculation: {e}", exc_info=True)
        # Set all results to None in case of a general download error
        for ticker in tickers:
            volatility_results[ticker] = None

    logger.info(f"Volatility calculation complete: {volatility_results}")
    return volatility_results

def check_volatility_signal(tickers: List[str], volatility_data: Dict[str, Optional[float]], recent_period: str = "5d") -> pd.DataFrame:
    """
    Checks if the latest daily return for given tickers exceeds their historical volatility.

    Args:
        tickers (List[str]): A list of stock ticker symbols.
        volatility_data (Dict[str, Optional[float]]): A dictionary mapping tickers to their
                                                      pre-calculated daily standard deviation (sigma).
                                                      Should come from calculate_stock_volatility.
        recent_period (str): The period for fetching recent data to calculate the latest return
                             (e.g., "5d", "10d"). Defaults to "5d".

    Returns:
        pd.DataFrame: A DataFrame with tickers as index and columns:
                      - 'latest_return': The most recent daily return (float or None).
                      - 'sigma': The provided daily volatility (float or None).
                      - 'is_significant': Boolean indicating if abs(latest_return) > sigma.
                      - 'status': A message indicating success or failure reason.
    """
    signal_results_list = [] # Store results as a list of dicts
    logger.info(f"Checking volatility signal for tickers: {tickers}")

    if not tickers:
        logger.warning("No tickers provided for signal check.")
        return pd.DataFrame(columns=['latest_return', 'sigma', 'is_significant', 'status']).rename_axis('ticker')

    # Pre-populate results for missing volatility data
    missing_vol_tickers = [t for t in tickers if t not in volatility_data or volatility_data.get(t) is None]
    for ticker in missing_vol_tickers:
        signal_results_list.append({'ticker': ticker, 'latest_return': None, 'sigma': None, 'is_significant': False, 'status': 'Missing volatility data'})

    # Tickers we need to fetch data for
    tickers_to_fetch = [t for t in tickers if t not in missing_vol_tickers]

    if not tickers_to_fetch:
        logger.warning("All provided tickers were missing volatility data.")
        return pd.DataFrame.from_records(signal_results_list).set_index('ticker')

    try:
        # Fetch data for the last ~N trading days to be safe
        logger.info(f"Fetching recent data for period: {recent_period} for tickers: {tickers_to_fetch}")
        recent_data = yf.download(tickers_to_fetch, period=recent_period, progress=False)

        if recent_data.empty:
            logger.error(f"Failed to download recent data for signal check for tickers: {tickers_to_fetch}")
            for ticker in tickers_to_fetch:
                signal_results_list.append({'ticker': ticker, 'latest_return': None, 'sigma': volatility_data.get(ticker), 'is_significant': False, 'status': 'Failed to fetch recent data'})
            return pd.DataFrame.from_records(signal_results_list).set_index('ticker')

        # Get the 'Close' prices
        recent_close = recent_data['Close']

        # Handle single ticker case
        if isinstance(recent_close, pd.Series):
            recent_close = recent_close.to_frame(name=tickers_to_fetch[0])

        if len(recent_close) < 2:
            logger.warning("Not enough recent data points (need at least 2) to calculate the latest return.")
            for ticker in tickers_to_fetch:
                signal_results_list.append({'ticker': ticker, 'latest_return': None, 'sigma': volatility_data.get(ticker), 'is_significant': False, 'status': 'Insufficient recent data'})
            return pd.DataFrame.from_records(signal_results_list).set_index('ticker')

        # Calculate the latest daily return (percentage change from penultimate to latest day)
        latest_returns = recent_close.iloc[-1] / recent_close.iloc[-2] - 1

        for ticker in tickers_to_fetch:
            sigma = volatility_data.get(ticker) # Already checked that sigma exists
            latest_ret = latest_returns.get(ticker)
            status = 'Success'
            is_significant = False
            final_ret = None

            if latest_ret is None or pd.isna(latest_ret):
                status = 'Could not calculate latest return'
            elif sigma is None or pd.isna(sigma): # Should not happen due to pre-check, but safety first
                status = 'Missing volatility data'
            else:
                final_ret = float(latest_ret)
                is_significant = abs(final_ret) > sigma
                logger.debug(f"Signal check for {ticker}: Return={latest_ret:.4f}, Sigma={sigma:.4f}, Significant={is_significant}")

            signal_results_list.append({
                'ticker': ticker,
                'latest_return': final_ret,
                'sigma': float(sigma) if sigma is not None else None,
                'is_significant': bool(is_significant),
                'status': status
            })

    except Exception as e:
        logger.error(f"An error occurred during volatility signal check: {e}", exc_info=True)
        processed_tickers = {d['ticker'] for d in signal_results_list}
        for ticker in tickers_to_fetch:
            if ticker not in processed_tickers:
                 signal_results_list.append({'ticker': ticker, 'latest_return': None, 'sigma': volatility_data.get(ticker), 'is_significant': False, 'status': f'Error: {e}'})

    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame.from_records(signal_results_list)

    # Set index and ensure all original tickers are present
    if not results_df.empty:
        results_df = results_df.set_index('ticker')
        # Reindex to include any tickers that might have been missed due to errors, filling with Nones/False
        results_df = results_df.reindex(tickers)
        # Fill NaN status potentially introduced by reindexing
        results_df['status'] = results_df['status'].fillna('Processing Error')
        results_df['is_significant'] = results_df['is_significant'].fillna(False)

    else: # Handle case where list was empty
         results_df = pd.DataFrame(index=pd.Index(tickers, name='ticker'), columns=['latest_return', 'sigma', 'is_significant', 'status'])
         results_df['status'] = 'Processing Error'
         results_df['is_significant'] = False


    logger.info(f"Volatility signal check complete. Returning DataFrame.")
    return results_df

def calculate_portfolio_breakdown(portfolio_df: pd.DataFrame) -> Optional[Tuple[float, float, float, pd.DataFrame]]:
    """
    Analyzes a portfolio DataFrame to provide a breakdown of long/short positions,
    total values, and relative weights.

    Args:
        portfolio_df: pandas DataFrame containing portfolio positions.
                      Must contain a column named 'mktValue'.

    Returns:
        A tuple containing:
        - total_long_value (float): Sum of market values for all long positions.
        - total_short_value (float): Sum of absolute market values for all short positions.
        - total_portfolio_value (float): Sum of absolute market values for all positions.
        - analyzed_df (pd.DataFrame): Original DataFrame with added columns:
            - 'PositionType': 'Long', 'Short', or 'Flat'
            - 'AbsoluteValue': Absolute market value of the position.
            - 'RelativeWeight': Weight relative to total longs or total shorts.
        Or None if the input is invalid or calculations fail.
    """
    if not isinstance(portfolio_df, pd.DataFrame):
        logger.error(f"Error: Input must be a pandas DataFrame, got {type(portfolio_df)}")
        return None

    if 'mktValue' not in portfolio_df.columns:
        logger.error("Error: DataFrame must contain a 'mktValue' column.")
        return None

    # Work on a copy to avoid modifying the original DataFrame outside the function scope
    analyzed_df = portfolio_df.copy()

    # Ensure 'mktValue' is numeric, coerce errors, fill NaNs with 0 for calculation
    analyzed_df['mktValue'] = pd.to_numeric(analyzed_df['mktValue'], errors='coerce').fillna(0.0)

    # Calculate Absolute Value
    analyzed_df['AbsoluteValue'] = analyzed_df['mktValue'].abs()

    # Determine Position Type
    analyzed_df['PositionType'] = 'Flat' # Default
    analyzed_df.loc[analyzed_df['mktValue'] > 1e-9, 'PositionType'] = 'Long' # Use small tolerance
    analyzed_df.loc[analyzed_df['mktValue'] < -1e-9, 'PositionType'] = 'Short'

    # Calculate Total Values
    total_long_value = analyzed_df.loc[analyzed_df['PositionType'] == 'Long', 'mktValue'].sum()
    total_short_value = analyzed_df.loc[analyzed_df['PositionType'] == 'Short', 'AbsoluteValue'].sum()
    total_portfolio_value = analyzed_df['AbsoluteValue'].sum()

    # Calculate Relative Weights
    analyzed_df['RelativeWeight'] = 0.0 # Initialize

    # Avoid division by zero if there are no longs or no shorts
    if total_long_value > 1e-9:
        long_mask = analyzed_df['PositionType'] == 'Long'
        analyzed_df.loc[long_mask, 'RelativeWeight'] = analyzed_df.loc[long_mask, 'mktValue'] / total_long_value

    if total_short_value > 1e-9:
        short_mask = analyzed_df['PositionType'] == 'Short'
        analyzed_df.loc[short_mask, 'RelativeWeight'] = analyzed_df.loc[short_mask, 'AbsoluteValue'] / total_short_value

    logger.info(f"Portfolio Breakdown: Longs=${total_long_value:.2f}, Shorts=${total_short_value:.2f}, Total=${total_portfolio_value:.2f}")

    return total_long_value, total_short_value, total_portfolio_value, analyzed_df
