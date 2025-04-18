
import pandas as pd
import numpy as np

def generate_average_volatility_signals(price_series: pd.Series, volatility_window: int, sigma_threshold: float) -> pd.Series:
    """
    Generates trading signals based on comparing daily returns to average rolling volatility.

    Args:
        price_series (pd.Series): Series of prices for a single ticker.
        volatility_window (int): Rolling window size for volatility calculation.
        sigma_threshold (float): Threshold for signal generation (multiple of average sigma).

    Returns:
        pd.Series: Series of signals (1 for Buy, -1 for Sell, 0 for Hold) indexed by date.
    """
    if price_series.empty or len(price_series) <= volatility_window:
        return pd.Series(0, index=price_series.index, dtype=int) # Not enough data

    # 1. Calculate Daily Returns
    daily_returns = price_series.pct_change()

    # 2. Calculate Rolling Volatility (Standard Deviation of Returns)
    rolling_sigma = daily_returns.rolling(window=volatility_window).std()

    # 3. Calculate Average Rolling Volatility
    valid_rolling_sigma = rolling_sigma.dropna()
    if valid_rolling_sigma.empty:
        # Not enough data after rolling calculation to determine average volatility
        average_rolling_sigma = np.nan
    else:
        average_rolling_sigma = valid_rolling_sigma.mean()

    # Handle case where average volatility couldn't be calculated
    if np.isnan(average_rolling_sigma) or average_rolling_sigma == 0:
         # If avg vol is NaN or zero, we cannot generate meaningful signals based on it. Hold.
         signals = pd.Series(0, index=price_series.index, dtype=int)
    else:
        # 4. Generate Signals based on Average Volatility
        buy_signal = daily_returns > (sigma_threshold * average_rolling_sigma)
        sell_signal = daily_returns < (-sigma_threshold * average_rolling_sigma)
        signals = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
        signals = pd.Series(signals, index=price_series.index, dtype=int)

    # Ensure the signal series has the same index as the input price series,
    # filling potential leading NaNs (from pct_change) with 0 (Hold).
    signals = signals.fillna(0).astype(int)


    return signals
