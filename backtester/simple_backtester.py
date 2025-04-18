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
                 interval: str = '1d'):
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

        # Align data with signals index
        backtest_data = self.data.loc[self.signals.index]

        for date, daily_signals in self.signals.iterrows():
            current_portfolio_value = self._cash
            prices_today = {}

            # Calculate current portfolio value based on yesterday's close/today's open logic proxy
            # For simplicity, we use today's close price from the data for valuation *before* trading
            for ticker in self.tickers:
                 close_price_col = f'{ticker}_Close'
                 if close_price_col in backtest_data.columns and date in backtest_data.index:
                     price = backtest_data.loc[date, close_price_col]
                     if pd.notna(price) and price > 0:
                         current_portfolio_value += self._positions[ticker] * price
                         prices_today[ticker] = price # Store price for trading

            self._portfolio_history.append((date, current_portfolio_value))

            # Execute trades based on signals for the current date
            for ticker in self.tickers:
                signal = daily_signals[ticker]
                trade_price = prices_today.get(ticker)

                if pd.isna(trade_price) or trade_price <= 0:
                    # print(f"Warning: No valid price for {ticker} on {date}. Skipping trade.")
                    continue # Skip if no valid price

                target_trade_value = current_portfolio_value * self.trade_size_fraction
                shares_to_trade = target_trade_value / trade_price

                # --- Buy Signal ---
                if signal == 1:
                    # Simple logic: If not already holding, buy. If holding, do nothing (or could add logic to increase position).
                    if self._positions[ticker] == 0:
                        if self._cash >= target_trade_value:
                             actual_shares = np.floor(shares_to_trade)
                             cost = actual_shares * trade_price
                             self._positions[ticker] += actual_shares
                             self._cash -= cost
                             self.trades.append({'Date': date, 'Ticker': ticker, 'Action': 'BUY',
                                                 'Shares': actual_shares, 'Price': trade_price, 'Cost': cost})
                             # print(f"{date}: BUY {actual_shares:.0f} {ticker} @ {trade_price:.2f}")
                        # else:
                            # print(f"{date}: Buy signal for {ticker}, but insufficient cash.")

                # --- Sell Signal ---
                elif signal == -1:
                     # Simple logic: If holding shares, sell all. If not holding, do nothing (or could add shorting logic).
                     if self._positions[ticker] > 0:
                         shares_held = self._positions[ticker]
                         proceeds = shares_held * trade_price
                         self._cash += proceeds
                         self._positions[ticker] = 0.0
                         self.trades.append({'Date': date, 'Ticker': ticker, 'Action': 'SELL',
                                              'Shares': shares_held, 'Price': trade_price, 'Cost': -proceeds})
                         # print(f"{date}: SELL {shares_held:.0f} {ticker} @ {trade_price:.2f}")

        self.portfolio = pd.DataFrame(self._portfolio_history, columns=['Date', 'PortfolioValue']).set_index('Date')
        print("Backtest simulation complete.")

    def calculate_performance(self, risk_free_rate: float = 0.0):
        """Calculates performance metrics for the backtest."""
        if self.portfolio is None or self.portfolio.empty:
            print("No portfolio data available. Run run_backtest() first.")
            return None

        print("Calculating performance metrics...")
        portfolio = self.portfolio['PortfolioValue']

        # 1. Total Return
        total_return = (portfolio.iloc[-1] / self.initial_capital) - 1

        # 2. Annualized Return
        years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 3. Portfolio Daily Returns
        daily_returns = portfolio.pct_change().dropna()

        # 4. Annualized Volatility (Standard Deviation)
        annualized_volatility = daily_returns.std() * np.sqrt(252) # Assuming 252 trading days

        # 5. Sharpe Ratio
        sharpe_ratio = ((annualized_return - risk_free_rate) / annualized_volatility
                        if annualized_volatility != 0 else np.nan)

        # 6. Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        # 7. Win Rate (optional, requires trade analysis)
        wins = sum(1 for trade in self.trades if trade['Action'] == 'SELL' and trade['Cost'] < 0) # Sell proceeds > buy cost implied
        losses = sum(1 for trade in self.trades if trade['Action'] == 'SELL' and trade['Cost'] >= 0) # Basic check
        total_closed_trades = wins + losses
        win_rate = wins / total_closed_trades if total_closed_trades > 0 else 0.0
        # Note: This win rate is simplified; a proper calculation needs to track entry/exit pairs.

        performance = {
            "Initial Capital": self.initial_capital,
            "Final Portfolio Value": portfolio.iloc[-1],
            "Total Return (%)": total_return * 100,
            "Annualized Return (%)": annualized_return * 100,
            "Annualized Volatility (%)": annualized_volatility * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown * 100,
            "Number of Trades": len(self.trades),
            "Win Rate (%)": win_rate * 100 # Simplified
        }

        print("Performance Metrics:")
        for key, value in performance.items():
            print(f"- {key}: {value:.2f}")

        return performance


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
        valid_trade_dates = [pd.Timestamp(trade['Date']) for trade in self.trades]
        valid_trades = [trade for trade in self.trades if pd.Timestamp(trade['Date']) in self.portfolio.index]
        buy_dates_trades = [pd.Timestamp(trade['Date']) for trade in valid_trades if trade['Action'] == 'BUY']
        sell_dates_trades = [pd.Timestamp(trade['Date']) for trade in valid_trades if trade['Action'] == 'SELL']

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
        interval='1d'
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
        interval='1d'
    )
    backtester_bb.run()
