import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import logging

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
