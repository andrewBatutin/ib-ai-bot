import pandas as pd
import numpy as np # Added for np.nan handling


def calculate_volume_bar_threshold(data_df: pd.DataFrame, target_bars_per_day: int, volume_col: str = 'Volume') -> float:
    """
    Estimates the volume threshold for generating volume bars aiming for a
    target number of bars per trading day.

    Based on the method described by Marcos López de Prado.

    Args:
        data_df (pd.DataFrame): DataFrame containing tick or high-frequency data.
                                 Must have a DatetimeIndex and a volume column.
        target_bars_per_day (int): The desired average number of volume bars per trading day.
        volume_col (str): The name of the column containing volume data. Defaults to 'Volume'.

    Returns:
        float: The estimated volume threshold. Returns np.nan if calculation fails
               (e.g., no data, no volume, no trading days).
    """
    if not isinstance(data_df.index, pd.DatetimeIndex):
        try:
            data_df.index = pd.to_datetime(data_df.index)
        except Exception as e:
            print(f"Error converting index to DatetimeIndex: {e}")
            return np.nan

    if data_df.empty:
        print("Error: Input DataFrame is empty.")
        return np.nan

    if volume_col not in data_df.columns:
        print(f"Error: Volume column '{volume_col}' not found in DataFrame.")
        return np.nan

    # Calculate number of unique trading days
    num_trading_days = data_df.index.normalize().nunique()

    if num_trading_days == 0:
        print("Error: Could not determine the number of trading days from the index.")
        return np.nan

    # Calculate Total Volume
    total_volume = data_df[volume_col].sum()

    if total_volume <= 0:
        print("Error: Total volume is zero or negative.")
        return np.nan

    # Calculate Average Daily Volume
    average_daily_volume = total_volume / num_trading_days

    # Calculate Threshold
    if target_bars_per_day <= 0:
        print("Error: target_bars_per_day must be positive.")
        return np.nan

    estimated_threshold = average_daily_volume / target_bars_per_day

    print(f"--- Threshold Calculation ---")
    print(f"Number of Trading Days: {num_trading_days}")
    print(f"Total Volume: {total_volume:,.0f}")
    print(f"Average Daily Volume: {average_daily_volume:,.0f}")
    print(f"Target Bars Per Day: {target_bars_per_day}")
    print(f"Estimated Volume Threshold: {estimated_threshold:,.0f}")
    print(f"---------------------------")

    return estimated_threshold