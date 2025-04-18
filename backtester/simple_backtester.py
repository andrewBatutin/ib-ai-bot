# backtester/simple_backtester.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

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

    def __init__(self, tickers: list, start_date: str, end_date: str,
                 initial_capital: float = 100000.0,
                 volatility_window: int = 21, # ~1 trading month
                 sigma_threshold: float = 1.0,
                 trade_size_fraction: float = 0.1,
                 interval: str = '1d'): # Fraction of capital per trade
        """
        Initializes the SimpleBacktester.

        Args:
            tickers (list): List of stock ticker symbols.
            start_date (str): Start date for backtesting (YYYY-MM-DD).
            end_date (str): End date for backtesting (YYYY-MM-DD).
            initial_capital (float): Starting capital for the backtest.
            volatility_window (int): Rolling window size (in days) for volatility calculation.
            sigma_threshold (float): The multiplier for sigma to trigger a signal.
            trade_size_fraction (float): Fraction of current portfolio value to allocate per trade.
            interval (str): Data interval for yfinance download (e.g., '1d', '1h', '5m').
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty.")

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.volatility_window = volatility_window
        self.sigma_threshold = sigma_threshold
        self.trade_size_fraction = trade_size_fraction
        self.interval = interval

        self.data = None # DataFrame to store historical price data
        self.signals = None # DataFrame to store trading signals
        self.portfolio = None # DataFrame to track portfolio value and positions
        self.trades = [] # List to store details of executed trades

        # Internal state for simulation
        self._cash = self.initial_capital
        self._positions = {ticker: 0.0 for ticker in self.tickers} # Shares held
        self._portfolio_history = [] # List of (date, portfolio_value) tuples

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
        """Calculates rolling volatility and generates trading signals."""
        if self.data is None or self.data.empty:
            print("No data available. Run _fetch_data() first.")
            return

        print("Calculating signals...")
        self.signals = pd.DataFrame(index=self.data.index)

        for ticker in self.tickers:
            price_series = self.data[ticker]

            # 1. Calculate Daily Returns
            daily_returns = price_series.pct_change()

            # 2. Calculate Rolling Volatility (Standard Deviation of Returns)
            rolling_sigma = daily_returns.rolling(window=self.volatility_window).std()

            # 3. Calculate Average Rolling Volatility
            # Calculate the mean of the rolling volatility series *once* per ticker
            # We need to handle the initial NaNs before calculating the mean
            valid_rolling_sigma = rolling_sigma.dropna()
            if valid_rolling_sigma.empty:
                average_rolling_sigma = np.nan # or some default, or skip signals for this ticker
            else:
                average_rolling_sigma = valid_rolling_sigma.mean()

            # 4. Generate Signals based on Average Volatility
            # Condition 1: Return > threshold * AVERAGE sigma (Buy)
            # Condition 2: Return < -threshold * AVERAGE sigma (Sell)
            # No shift needed here as average_rolling_sigma is constant for the ticker
            buy_signal = daily_returns > (self.sigma_threshold * average_rolling_sigma)
            sell_signal = daily_returns < (-self.sigma_threshold * average_rolling_sigma)

            # Combine signals: 1 for Buy, -1 for Sell, 0 for Hold
            self.signals[ticker] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

        # Remove initial period where rolling window is not yet full
        self.signals = self.signals.iloc[self.volatility_window:]
        print("Signals calculated.")
        # print(self.signals.head()) # Optional: Print head


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
        """Plots the portfolio value over time."""
        if self.portfolio is None or self.portfolio.empty:
            print("No portfolio data to plot. Run run_backtest() first.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio.index, self.portfolio['PortfolioValue'], label='Portfolio Value')

        # Optional: Add buy/sell markers
        buy_dates = [trade['Date'] for trade in self.trades if trade['Action'] == 'BUY']
        sell_dates = [trade['Date'] for trade in self.trades if trade['Action'] == 'SELL']

        if buy_dates:
             buy_values = self.portfolio.loc[buy_dates, 'PortfolioValue']
             plt.scatter(buy_values.index, buy_values.values, marker='^', color='g', label='Buy Signal', alpha=0.7, s=100)
        if sell_dates:
             sell_values = self.portfolio.loc[sell_dates, 'PortfolioValue']
             plt.scatter(sell_values.index, sell_values.values, marker='v', color='r', label='Sell Signal', alpha=0.7, s=100)


        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
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
    tickers_to_test = ['AAPL', 'MSFT']#, 'GOOGL'] # Example tickers
    start = '2021-01-01'
    end = '2023-12-31'

    backtester = SimpleBacktester(
        tickers=tickers_to_test,
        start_date=start,
        end_date=end,
        initial_capital=10000.0,
        volatility_window=21, # ~1 month
        sigma_threshold=1.5, # Trade if return > 1.5 * sigma
        trade_size_fraction=0.2, # Use 20% of portfolio value per trade
        interval='5m' # Use 5-minute interval
    )

    backtester.run()
    backtester.plot_results() # Show plot after running

    # Access results if needed
    # performance_results = backtester.calculate_performance()
    # trades_log = pd.DataFrame(backtester.trades)
    # print("\nTrade Log:")
    # print(trades_log)
