import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def plot_price_with_cusum_events(price_series, events, ticker_name, threshold):
    """
    Plots a price series and marks CUSUM filter events with vertical lines.

    Args:
        price_series (pd.Series): Series with DatetimeIndex and prices.
        events (pd.DatetimeIndex): Timestamps of the CUSUM events.
        ticker_name (str): Name of the ticker for the title.
        threshold (float): CUSUM threshold used, for the legend.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the price series
    ax.plot(price_series.index, price_series.values, label='Price', color='blue', linewidth=1)

    # Add vertical lines for CUSUM events
    event_count = 0
    for event_time in events:
        if event_time in price_series.index: # Check if event time is in the data
            ax.axvline(event_time, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                       label='_nolegend_' if event_count > 0 else f'CUSUM Event (th={threshold})') # Label only the first line
            event_count += 1
        else:
            print(f"Warning: CUSUM event time {event_time} not found in price series index. Skipping.")

    if event_count == 0:
         print("Warning: No CUSUM events were plotted (timestamps might not align exactly with price data index).")


    # Formatting
    ax.set_title(f"{ticker_name} Price and CUSUM Filter Events")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Format x-axis dates
    major_locator = mdates.AutoDateLocator()
    major_formatter = mdates.ConciseDateFormatter(major_locator)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_bars_matplotlib(df, title, ticker_name="Ticker"):
    """
    Plots OHLCV data using Matplotlib directly.

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex and columns
                           'open', 'high', 'low', 'close', 'volume'.
        title (str): The title for the chart.
        ticker_name (str): Name of the ticker for the title.
    """
    if df is None or df.empty:
        print(f"Plotting skipped for '{title}': DataFrame is None or empty.")
        return

    # Standardize column names and check for required ones
    df.columns = df.columns.str.lower()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Plotting skipped for '{title}': Missing required OHLCV columns. Columns found: {df.columns.tolist()}")
        return

    # Convert to numeric and drop NaNs
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)

    if df.empty:
        print(f"Plotting skipped for '{title}': No valid data after type conversion/dropna.")
        return

    # --- Plotting Setup ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]}) # Price plot gets more space
    ax_price = axes[0]
    ax_volume = axes[1]

    # Use numerical indices for plotting shapes, then format x-axis with dates
    x_indices = range(len(df.index))
    x_dates = df.index # Keep original dates for formatting

    # Define colors
    color_up = 'green'
    color_down = 'red'

    # --- Price Plot (Candlesticks) ---
    bar_width = 0.6 # Width of the candle body
    wick_width = 0.1 # Width of the wick line

    for i in x_indices:
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]

        if close_price >= open_price:
            color = color_up
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = color_down
            body_bottom = close_price
            body_height = open_price - close_price

        # Draw the wick (high-low line)
        ax_price.vlines(i, low_price, high_price, color=color, linewidth=wick_width)

        # Draw the body (open-close rectangle)
        # Only draw body if open != close, otherwise it's a thin line
        if body_height > 0:
             body = Rectangle((i - bar_width / 2, body_bottom), bar_width, body_height,
                              facecolor=color, edgecolor=color, zorder=3) # zorder puts body on top
             ax_price.add_patch(body)
        else:
             # Draw a horizontal line if open == close
             ax_price.hlines(open_price, i - bar_width / 2, i + bar_width / 2, color=color, linewidth=1)


    # --- Volume Plot ---
    volume_colors = [color_up if df['close'].iloc[i] >= df['open'].iloc[i] else color_down for i in x_indices]
    ax_volume.bar(x_indices, df['volume'], width=bar_width, color=volume_colors)

    # --- Formatting ---
    ax_price.set_title(f"{ticker_name} - {title}")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle='--', alpha=0.6)

    ax_volume.set_ylabel("Volume")
    ax_volume.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis to show dates
    # Use a locator and formatter for better date display
    # Adjust the locator/formatter based on the time range of your data
    major_locator = mdates.AutoDateLocator() # Sensible tick placement
    major_formatter = mdates.ConciseDateFormatter(major_locator) # Nice date format

    ax_volume.xaxis.set_major_locator(major_locator)
    ax_volume.xaxis.set_major_formatter(major_formatter)

    # Improve layout and rotate date labels
    fig.autofmt_xdate() # Auto-rotate date labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly to prevent title overlap
    plt.subplots_adjust(hspace=0.1) # Reduce space between plots
