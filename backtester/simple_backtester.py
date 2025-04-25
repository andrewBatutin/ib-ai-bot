# backtester/simple_backtester.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt # Ensure pyplot is imported

# Import the strategy functions
from .strategies import generate_average_volatility_signals, generate_bollinger_bands_signals

class SimpleBacktester:
    """
    A simple backtester for a volatility-based trading strategy.

    Strategy:
    - Calculate rolling historical volatility (sigma) over a defined lookback period.
    - Generate a buy signal if the daily return exceeds `+threshold * sigma`.
    - Generate a sell signal if the daily return falls below `-threshold * sigma`.
    - Hold otherwise.
    - Assumes trading at the closing price of the signal day.
    """

    def __init__(self, tickers: list[str],
                 start_date: str, end_date: str,
                 strategy_name: str,
                 strategy_params: dict,
                 initial_capital: float = 100000.0,
                 trade_size_fraction: float = 0.1,
                 interval: str = '1d',
                 commission_per_trade: float = 0.0):
        """
        Initializes the SimpleBacktester.

        Args:
            tickers (list): List of ticker symbols.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            strategy_name (str): Name of the strategy to use (e.g., 'average_volatility', 'bollinger_bands').
            strategy_params (dict): Dictionary of parameters for the chosen strategy.
                                    Example for 'average_volatility': {'volatility_window': 21, 'sigma_threshold': 1.5}
                                    Example for 'bollinger_bands': {'window': 20, 'num_std': 2.0}
            initial_capital (float, optional): Starting capital. Defaults to 100000.
            trade_size_fraction (float, optional): Fraction of portfolio value per trade. Defaults to 0.1.
            interval (str, optional): yfinance data interval. Defaults to '1d'.
            commission_per_trade (float, optional): Fixed commission cost per trade. Defaults to 0.0.
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty.")
        if not isinstance(tickers, list):
            tickers = [tickers]

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params
        self.initial_capital = initial_capital
        self.trade_size_fraction = trade_size_fraction
        self.interval = interval
        self.commission_per_trade = commission_per_trade # Store commission

        # Validate strategy parameters based on name
        self._validate_strategy_params()

        self.data = None # DataFrame to store historical price data
        self.signals = None # DataFrame to store trading signals
        self.portfolio = None # DataFrame to track portfolio value and positions
        self.trades = [] # List to store trade details

        # Internal state for simulation
        self._cash = self.initial_capital
        self._positions = {ticker: 0.0 for ticker in self.tickers} # Shares held
        self._portfolio_history = [] # List of (date, portfolio_value) tuples
        self._average_cost = {ticker: 0.0 for ticker in self.tickers} # Initialize average cost dictionary

    def _validate_strategy_params(self):
        """Validates if the required parameters are present for the selected strategy."""
        required_params = {}
        if self.strategy_name == 'average_volatility':
            required_params = {'volatility_window', 'sigma_threshold'}
        elif self.strategy_name == 'bollinger_bands':
            required_params = {'window', 'num_std'}
        else:
            raise ValueError(f"Unknown strategy name: {self.strategy_name}")

        missing_params = required_params - set(self.strategy_params.keys())
        if missing_params:
            raise ValueError(f"Missing parameters for strategy '{self.strategy_name}': {missing_params}")

    def _fetch_data(self):
        """Fetches historical adjusted closing prices using yfinance."""
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date} with interval {self.interval}...")
        try:
            # Fetch Adj Close for returns, Close for trade execution price
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, interval=self.interval, prepost=True)
            if data.empty:
                raise ValueError("No data fetched. Check tickers and date range.")

            # Use Adj Close for calculations if available, else Close
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            if price_col not in data.columns:
                 raise ValueError(f"Could not find '{price_col}' in fetched data.")

            # Handle single vs multiple tickers
            if len(self.tickers) == 1:
                self.data = data[[price_col]].rename(columns={price_col: self.tickers[0]})
                self.data[f'{self.tickers[0]}_Close'] = data['Close'] # Keep Close price for trading
            else:
                self.data = data[price_col]
                 # Add Close prices explicitly for trading
                for ticker in self.tickers:
                    self.data[f'{ticker}_Close'] = data['Close'][ticker]

            # Drop rows with any NaN values (often at the beginning)
            self.data.dropna(inplace=True)

            if self.data.empty:
                 raise ValueError("Data became empty after dropping NaNs.")

            print("Data fetched successfully.")
            # print(self.data.head()) # Optional: Print head for verification

        except Exception as e:
            print(f"Error fetching data: {e}")
            self.data = None


    def _calculate_signals(self):
        """Calculates trading signals using the selected strategy."""
        if self.data is None or self.data.empty:
            print("No data available. Run _fetch_data() first.")
            return

        print(f"Calculating signals using '{self.strategy_name}' strategy...")
        self.signals = pd.DataFrame(index=self.data.index)

        for ticker in self.tickers:
            price_col_base = ticker if len(self.tickers) == 1 else ticker
            if price_col_base in self.data.columns:
                price_series = self.data[price_col_base]
            else:
                print(f"Warning: Price column '{price_col_base}' not found directly for {ticker}. Skipping signal calculation.")
                self.signals[ticker] = 0 # Assign hold signal if price data is missing
                continue

            # Dynamically call the selected strategy function
            if self.strategy_name == 'average_volatility':
                ticker_signals = generate_average_volatility_signals(
                    price_series=price_series,
                    volatility_window=self.strategy_params['volatility_window'],
                    sigma_threshold=self.strategy_params['sigma_threshold']
                )
            elif self.strategy_name == 'bollinger_bands':
                ticker_signals = generate_bollinger_bands_signals(
                    price_series=price_series,
                    window=self.strategy_params['window'],
                    num_std=self.strategy_params['num_std']
                )
            else:
                # This should not happen due to validation in __init__, but added for safety
                print(f"Error: Strategy '{self.strategy_name}' function not implemented in _calculate_signals.")
                ticker_signals = pd.Series(0, index=price_series.index, dtype=int)

            self.signals[ticker] = ticker_signals

        print("Signals calculated.")


    def _execute_trade(self, date, ticker, side, shares, price):
        """Executes a single trade and updates portfolio state, including average cost and PnL."""
        # Ensure shares and price are valid numbers
        if not isinstance(shares, (int, float)) or not isinstance(price, (int, float)) or pd.isna(shares) or pd.isna(price) or shares <= 1e-9 or price <= 0:
             print(f"Warning: Invalid shares ({shares}) or price ({price}) for trade on {date} for {ticker}. Skipping.")
             return

        pnl = 0.0 # Initialize PnL for this trade

        if side == 'BUY':
            trade_cost = shares * price
            commission = self.commission_per_trade
            total_cost = trade_cost + commission
            if self._cash >= total_cost:
                # Update average cost *before* changing position
                current_position = self._positions[ticker]
                current_total_cost = self._average_cost[ticker] * current_position
                new_position = current_position + shares

                if new_position > 1e-9: # Avoid division by zero
                    new_total_cost = current_total_cost + trade_cost
                    self._average_cost[ticker] = new_total_cost / new_position
                else:
                    self._average_cost[ticker] = 0 # Reset if position becomes zero (unlikely on buy)

                self._cash -= total_cost
                self._positions[ticker] = new_position
                self.trades.append({
                    'date': date, 'ticker': ticker, 'side': side,
                    'shares': shares, 'price': price, 'trade_value': trade_cost,
                    'avg_cost_after': self._average_cost[ticker], 'pnl': pnl, # PnL is 0 for buy
                    'commission': commission
                })
                # print(f"{date}: BOUGHT {shares:.4f} {ticker} @ {price:.2f}") # Optional log
            else:
                print(f"Warning: Insufficient cash ({self._cash:.2f}) to buy {shares:.4f} {ticker} @ {price:.2f} (Cost: {trade_cost:.2f}, Commission: {commission:.2f}) on {date}. Skipping trade.")
        elif side == 'SELL':
            # Ensure we have enough shares to sell (allow for float precision issues)
            # Sell at most the shares we currently hold
            shares_to_sell = min(shares, self._positions[ticker])

            if shares_to_sell > 1e-9: # Check if there's a meaningful amount to sell
                trade_proceeds = shares_to_sell * price

                # Calculate PnL based on the average cost of the shares being sold
                if self._average_cost[ticker] > 0: # Ensure we have a valid average cost
                    cost_of_goods_sold = self._average_cost[ticker] * shares_to_sell
                    pnl = trade_proceeds - cost_of_goods_sold
                else:
                     pnl = 0.0 # PnL is 0 if selling without a tracked buy/cost basis
                     # print(f"Warning: Selling {shares_to_sell:.4f} {ticker} on {date} without a recorded average cost.")

                commission = self.commission_per_trade
                self._cash += trade_proceeds - commission
                self._positions[ticker] -= shares_to_sell

                # Reset average cost if position is closed (or very close to zero)
                if abs(self._positions[ticker]) < 1e-9:
                    self._positions[ticker] = 0.0 # Clean up small residuals
                    self._average_cost[ticker] = 0.0 # Reset average cost

                self.trades.append({
                    'date': date, 'ticker': ticker, 'side': side,
                    'shares': shares_to_sell, 'price': price, 'trade_value': trade_proceeds,
                    'avg_cost_after': self._average_cost[ticker], 'pnl': pnl, # Record realized PnL
                    'commission': commission
                })
                # print(f"{date}: SOLD {shares_to_sell:.4f} {ticker} @ {price:.2f}, PnL: {pnl:.2f}") # Optional log
            # else: # If shares_to_sell is negligible or negative, don't execute
            #    if shares > 1e-9: # Only warn if the *requested* sell was significant
            #         print(f"Warning: Attempted to sell {shares:.4f} {ticker} on {date}, but holding {self._positions[ticker]:.4f}. Selling {shares_to_sell:.4f}.")


    def run_backtest(self):
        """Runs the backtest simulation based on generated signals."""
        if self.signals is None or self.signals.empty:
            print("No signals available. Run _calculate_signals() first.")
            return

        print("Running backtest simulation...")
        self._cash = self.initial_capital
        self._positions = {ticker: 0.0 for ticker in self.tickers}
        self._portfolio_history = []
        self.trades = []

        # Align data with signals index - Ensure data covers the signal period + 1 for execution
        # We need prices for the day *after* the signal is generated
        min_signal_date = self.signals.index.min()
        max_signal_date = self.signals.index.max()

        # Ensure data includes the day after the last signal for execution
        # Note: This might require fetching slightly more data initially if end_date was tight
        backtest_data = self.data.loc[min_signal_date:]

        # --- FIX: Shift signals by 1 period to avoid lookahead bias ---
        # The signal generated at time `t` (based on data up to `t`) is used for trading at time `t+1`
        signals_shifted = self.signals.shift(1).dropna() # Shift and remove the first row (NaN)

        # Iterate through the *shifted* signals index
        for date in signals_shifted.index:
            # Ensure the current date exists in our price data (it should, as we shifted)
            if date not in backtest_data.index:
                # print(f"Warning: Price data missing for execution date {date}. Skipping.") # Optional warning
                continue

            # --- Portfolio valuation *before* trading on 'date' ---
            # Use prices from the *previous* available index for valuation before trade
            # Find the index location for 'date' and get the previous one
            current_date_loc = backtest_data.index.get_loc(date)
            if current_date_loc == 0:
                 # Cannot value on the very first day before trading, use initial capital
                 current_portfolio_value = self._cash
                 # Store initial portfolio value using the first shifted signal date
                 if not self._portfolio_history: # Store only once
                     self._portfolio_history.append((date, current_portfolio_value))
            else:
                 prev_date = backtest_data.index[current_date_loc - 1]
                 current_portfolio_value = self._cash
                 for ticker in self.tickers:
                     close_price_col = f'{ticker}_Close'
                     # Use previous day's close for valuation before today's trades
                     if close_price_col in backtest_data.columns and prev_date in backtest_data.index:
                         price = backtest_data.loc[prev_date, close_price_col]
                         if pd.notna(price) and price > 0:
                             current_portfolio_value += self._positions[ticker] * price
                 # Store portfolio value *before* trades for this period
                 self._portfolio_history.append((date, current_portfolio_value))


            # --- Execute trades based on *yesterday's* signal, using *today's* price ---
            daily_signals = signals_shifted.loc[date]
            prices_for_trade = {} # Get actual trading prices for 'date'

            for ticker in self.tickers:
                 close_price_col = f'{ticker}_Close'
                 if close_price_col in backtest_data.columns and date in backtest_data.index:
                     price = backtest_data.loc[date, close_price_col]
                     if pd.notna(price) and price > 0:
                          prices_for_trade[ticker] = price

            for ticker in self.tickers:
                signal = daily_signals[ticker]
                trade_price = prices_for_trade.get(ticker) # Use today's price

                if pd.isna(trade_price) or trade_price <= 0:
                    # print(f"Warning: Valid trade price not available for {ticker} on {date}. Skipping trade.") # Optional
                    continue # Skip trade if no valid price

                # Determine desired position based on signal & current value
                # Use the *pre-trade* portfolio value calculated earlier for sizing
                target_trade_value = current_portfolio_value * self.trade_size_fraction
                target_shares = target_trade_value / trade_price if trade_price > 0 else 0

                current_shares = self._positions[ticker]

                # Simplified logic: If buy signal and not already long -> buy target shares
                # If sell signal and currently long -> sell all shares held
                if signal == 1 and current_shares <= 1e-9: # Buy condition (use tolerance)
                     if target_shares > 0: # Ensure calculated shares are positive
                         # Use floor/round for share calculation if needed, here using fractional
                         self._execute_trade(date, ticker, 'BUY', target_shares, trade_price)
                elif signal == -1 and current_shares > 1e-9: # Sell condition (use tolerance)
                    # Selling all current shares, not based on target_shares for sell signal
                    self._execute_trade(date, ticker, 'SELL', current_shares, trade_price)
                # Add logic for short selling if needed based on signal == -1 and current_shares <= 1e-9

        # Final portfolio calculation after the loop
        # Use the last recorded date in portfolio history for final valuation basis
        if self._portfolio_history:
            last_recorded_date = self._portfolio_history[-1][0]
            # Find the latest data point >= last_recorded_date for final price
            final_valuation_data_points = backtest_data[backtest_data.index >= last_recorded_date]
            if not final_valuation_data_points.empty:
                final_valuation_date = final_valuation_data_points.index[0]

                final_portfolio_value = self._cash
                for ticker in self.tickers:
                    close_price_col = f'{ticker}_Close'
                    if close_price_col in backtest_data.columns and final_valuation_date in backtest_data.index:
                        price = backtest_data.loc[final_valuation_date, close_price_col]
                        if pd.notna(price) and price > 0:
                            final_portfolio_value += self._positions[ticker] * price
                        else: # Fallback to last known price if final price is NaN
                           last_valid_price_index = backtest_data[close_price_col].last_valid_index()
                           if last_valid_price_index is not None:
                                price = backtest_data.loc[last_valid_price_index, close_price_col]
                                if pd.notna(price) and price > 0:
                                     final_portfolio_value += self._positions[ticker] * price
                # Append final value using the date it was calculated for
                self._portfolio_history.append((final_valuation_date, final_portfolio_value))
            else: # If no data after last recorded date, use the last recorded value
                 pass # The last value is already in history

        elif not backtest_data.empty:
             # Handle case where signals were generated but all NaN after shift
             final_portfolio_value = self._cash # Only cash
             # Use last data point available if possible
             self._portfolio_history.append((backtest_data.index[-1], final_portfolio_value))


        # Convert portfolio history to DataFrame
        if self._portfolio_history:
            self.portfolio = pd.DataFrame(self._portfolio_history, columns=['Date', 'PortfolioValue']).set_index('Date')
            # Ensure index is unique, keep last entry if duplicates exist
            self.portfolio = self.portfolio[~self.portfolio.index.duplicated(keep='last')]
            # Ensure index is datetime
            self.portfolio.index = pd.to_datetime(self.portfolio.index)
            # Ensure index is sorted
            self.portfolio = self.portfolio.sort_index()
            self.portfolio['returns'] = self.portfolio['PortfolioValue'].pct_change().fillna(0)
        else:
            # Create an empty DataFrame with expected columns
            self.portfolio = pd.DataFrame(columns=['PortfolioValue', 'returns'])
            self.portfolio.index.name = 'Date'
            self.portfolio.index = pd.to_datetime(self.portfolio.index)

        print("Backtest simulation finished.")

    def calculate_performance(self, risk_free_rate: float = 0.0):
        """Calculates performance metrics for the backtest.

        Args:
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation.

        Returns:
            dict: A dictionary containing key performance metrics.
        """
        if self.portfolio is None or self.portfolio.empty or len(self.portfolio) < 2:
            print("Portfolio data not available or insufficient for performance calculation (requires >= 2 data points).")
            # Return default metrics
            metrics = {
                'Initial Capital': self.initial_capital,
                'Final Portfolio Value': self._cash if self.portfolio is None or self.portfolio.empty else self.portfolio['PortfolioValue'].iloc[-1],
                'Total Return (%)': ((self._cash / self.initial_capital - 1) * 100
                                   if self.initial_capital > 0 else 0.0),
                'Annualized Return (%)': 0.0,
                'Annualized Volatility (%)': 0.0,
                'Sharpe Ratio': np.nan,
                'Max Drawdown (%)': 0.0,
                'Number of Trades': len(self.trades),
                'Win Rate (%)': np.nan
            }
            # If portfolio exists but has only one row, calculate final value/return based on that
            if self.portfolio is not None and len(self.portfolio) == 1:
                 metrics['Final Portfolio Value'] = self.portfolio['PortfolioValue'].iloc[-1]
                 metrics['Total Return (%)'] = ((metrics['Final Portfolio Value'] / self.initial_capital - 1) * 100
                                              if self.initial_capital > 0 else 0.0)
            return metrics

        print("Calculating performance metrics...")

        # Basic metrics
        final_value = self.portfolio['PortfolioValue'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100 if self.initial_capital > 0 else 0.0

        # Time-based calculations
        time_delta = self.portfolio.index[-1] - self.portfolio.index[0]
        min_year_fraction = 1.0 / (365.25 * 24 * 60 * 60) # Min duration = 1 second in years
        years = max(time_delta.total_seconds() / (365.25 * 24 * 60 * 60), min_year_fraction)

        # Annualized Return (Geometric)
        if self.initial_capital > 0:
            base_ratio = max(final_value / self.initial_capital, 0)
            annualized_return = (base_ratio ** (1 / years) - 1) * 100
        else:
             annualized_return = 0.0

        # Portfolio returns
        portfolio_returns = self.portfolio['PortfolioValue'].pct_change().dropna()
        if isinstance(portfolio_returns, float):
            portfolio_returns = pd.Series([portfolio_returns], index=[self.portfolio.index[1]])

        if portfolio_returns.empty:
             annualized_volatility = 0.0
             sharpe_ratio = np.nan
             max_drawdown = 0.0
        else:
            # --- Dynamic Annualization for Volatility/Sharpe ---
            trading_periods_per_year = 252 # Default
            inferred_freq = pd.infer_freq(self.portfolio.index)
            if inferred_freq:
                freq_map = {
                    'B': 252, 'D': 365, 'W': 52, 'M': 12, 'MS': 12,
                    'Q': 4, 'QS': 4, 'A': 1, 'Y': 1, 'AS': 1, 'YS': 1,
                    'H': 252 * 6.5, 'T': 252 * 6.5 * 60, 'min': 252 * 6.5 * 60,
                    'S': 252 * 6.5 * 60 * 60, 'L': 252 * 6.5 * 60 * 60 * 1000, 'ms': 252 * 6.5 * 60 * 60 * 1000,
                    'U': 252 * 6.5 * 60 * 60 * 1000 * 1000, 'us': 252 * 6.5 * 60 * 60 * 1000 * 1000
                }
                freq_str_norm = inferred_freq.split('-')[0].upper()
                if freq_str_norm in freq_map:
                    trading_periods_per_year = freq_map[freq_str_norm]
                else:
                    # Try interval calculation
                    avg_interval_seconds = portfolio_returns.index.to_series().diff().mean().total_seconds()
                    if pd.notna(avg_interval_seconds) and avg_interval_seconds > 0:
                        seconds_in_trading_year = 252 * 6.5 * 60 * 60
                        trading_periods_per_year = seconds_in_trading_year / avg_interval_seconds
                        print(f"Note: Annualizing based on inferred average interval ({avg_interval_seconds:.2f}s). Assumes 6.5h/day, 252 days/yr.")
                    else:
                        print("Warning: Could not determine trading periods per year from interval. Using 252.")
                        trading_periods_per_year = 252
            else:
                print("Warning: Could not infer data frequency. Assuming 252 trading periods/year.")
                trading_periods_per_year = 252

            trading_periods_per_year = max(trading_periods_per_year, 1)

            # Annualized Volatility
            volatility = portfolio_returns.std()
            if pd.isna(volatility) or volatility < 1e-9:
                annualized_volatility = 0.0
            else:
                annualized_volatility = volatility * np.sqrt(trading_periods_per_year) * 100

            # Sharpe Ratio
            if annualized_volatility > 1e-6:
                annualized_mean_return_pct = annualized_return # Use geometric annualized return
                sharpe_ratio = (annualized_mean_return_pct - (risk_free_rate * 100)) / annualized_volatility
            else:
                sharpe_ratio = np.nan # Undefined if volatility is zero

            # Max Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0


        # Trade-based metrics
        win_rate = np.nan
        num_closing_trades = 0
        total_trades = len(self.trades)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            # Consider only closing trades (SELL in long-only) with valid PnL
            closing_trades = trades_df[(trades_df['side'] == 'SELL') & trades_df['pnl'].notna()]
            num_closing_trades = len(closing_trades)
            if num_closing_trades > 0:
                # A win is a closing trade with positive PnL (use tolerance)
                winning_trades_count = len(closing_trades[closing_trades['pnl'] > 1e-9])
                win_rate = (winning_trades_count / num_closing_trades) * 100

        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Portfolio Value': final_value,
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Number of Trades': total_trades, # Total buys and sells executed
            'Win Rate (%)': win_rate # Based on closing trades with realized PnL
        }

        print("--- Performance Metrics ---")
        for key, value in metrics.items():
            if isinstance(value, (float, np.number)):
                 print(f"- {key}: {value:.2f}")
            else:
                 print(f"- {key}: {value}")
        print("-------------------------")

        return metrics


    def plot_results(self):
        """Plots the portfolio value over time and individual security prices with signals."""
        if self.portfolio is None or self.portfolio.empty:
            print("No portfolio data to plot. Run run_backtest() first.")
            return
        if self.data is None or self.data.empty:
            print("No price data available for plotting security charts.")
            return
        if self.signals is None or self.signals.empty:
            print("No signal data available for plotting security charts.")
            return

        num_tickers = len(self.tickers)
        # Create a figure with plots: 1 for portfolio + 1 for each ticker
        fig, axes = plt.subplots(num_tickers + 1, 1, figsize=(12, 6 * (num_tickers + 1)), sharex=True)

        # Ensure axes is always a list-like object, even if only one subplot
        if num_tickers == 0:
            ax_portfolio = axes
        else:
            ax_portfolio = axes[0]
            ax_tickers = axes[1:]

        # 1. Plot Portfolio Value
        ax_portfolio.plot(self.portfolio.index, self.portfolio['PortfolioValue'], label='Portfolio Value')
        # Optional: Add buy/sell markers based on actual trades on portfolio plot
        # Filter trades to match portfolio index dates if necessary
        valid_trade_dates = [pd.Timestamp(trade['date']) for trade in self.trades]
        valid_trades = [trade for trade in self.trades if pd.Timestamp(trade['date']) in self.portfolio.index]
        buy_dates_trades = [pd.Timestamp(trade['date']) for trade in valid_trades if trade['side'] == 'BUY']
        sell_dates_trades = [pd.Timestamp(trade['date']) for trade in valid_trades if trade['side'] == 'SELL']

        if buy_dates_trades:
            buy_values = self.portfolio.loc[buy_dates_trades, 'PortfolioValue']
            ax_portfolio.scatter(buy_values.index, buy_values.values, marker='^', color='g', label='Executed Buy', alpha=0.7, s=100, zorder=5)
        if sell_dates_trades:
            sell_values = self.portfolio.loc[sell_dates_trades, 'PortfolioValue']
            ax_portfolio.scatter(sell_values.index, sell_values.values, marker='v', color='r', label='Executed Sell', alpha=0.7, s=100, zorder=5)

        ax_portfolio.set_title('Portfolio Value Over Time')
        ax_portfolio.set_ylabel('Portfolio Value ($)')
        ax_portfolio.legend()
        ax_portfolio.grid(True)

        # 2. Plot Individual Security Prices with Signals
        if num_tickers > 0:
            # Align signals index with data index if necessary (though backtest should handle this)
            signals_aligned = self.signals.reindex(self.data.index).fillna(0)

            for i, ticker in enumerate(self.tickers):
                ax = ax_tickers[i]
                price_col = f'{ticker}_Close' # Use the close price for plotting
                if price_col not in self.data.columns:
                    print(f"Warning: Close price column '{price_col}' not found for {ticker}. Skipping price plot.")
                    continue

                # Plot price
                ax.plot(self.data.index, self.data[price_col], label=f'{ticker} Price')

                # Find signal points
                buy_signals = signals_aligned[signals_aligned[ticker] == 1].index
                sell_signals = signals_aligned[signals_aligned[ticker] == -1].index

                # Plot signals on price chart if they exist in the data index
                valid_buy_signals = self.data.index.intersection(buy_signals)
                valid_sell_signals = self.data.index.intersection(sell_signals)

                if not valid_buy_signals.empty:
                    ax.scatter(valid_buy_signals, self.data.loc[valid_buy_signals, price_col],
                               marker='^', color='lime', label='Buy Signal', alpha=0.9, s=80, zorder=5)
                if not valid_sell_signals.empty:
                    ax.scatter(valid_sell_signals, self.data.loc[valid_sell_signals, price_col],
                               marker='v', color='red', label='Sell Signal', alpha=0.9, s=80, zorder=5)

                ax.set_title(f'{ticker} Price and Signals')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)

        plt.xlabel('Date') # Shared X-axis label
        plt.tight_layout()
        plt.show()


    def run(self):
        """Runs the full backtesting pipeline: fetch, signal, simulate, analyze."""
        self._fetch_data()
        if self.data is not None:
            self._calculate_signals()
            if self.signals is not None:
                self.run_backtest()
                if self.portfolio is not None:
                    self.calculate_performance()
                    # self.plot_results() # Uncomment to automatically plot


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Average Volatility Strategy
    print("--- Running Average Volatility Strategy ---")
    avg_vol_params = {
        'volatility_window': 21, # ~1 month
        'sigma_threshold': 1.5   # Trade if return > 1.5 * avg sigma
    }
    backtester_avg_vol = SimpleBacktester(
        tickers=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        strategy_name='average_volatility',
        strategy_params=avg_vol_params,
        initial_capital=10000.0,
        trade_size_fraction=0.2, # Use 20% of portfolio value per trade
        interval='1d',
        commission_per_trade=5.0
    )
    backtester_avg_vol.run()
    print("\n")

    # Example 2: Bollinger Bands Strategy
    print("--- Running Bollinger Bands Strategy ---")
    bb_params = {
        'window': 20,    # 20-day SMA
        'num_std': 2.0   # 2 standard deviations
    }
    backtester_bb = SimpleBacktester(
        tickers=['NVDA'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        strategy_name='bollinger_bands',
        strategy_params=bb_params,
        initial_capital=10000.0,
        trade_size_fraction=0.25,
        interval='1d',
        commission_per_trade=5.0
    )
    backtester_bb.run()
