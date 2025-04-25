import pandas as pd
import itertools
import numpy as np
import time
from .simple_backtester import SimpleBacktester # Use relative import

class GridSearchOptimizer:
    """
    Performs grid search optimization for SimpleBacktester strategies.
    """
    def __init__(self,
                 backtester_class,
                 fixed_params: dict,
                 param_grid: dict,
                 metric_to_optimize: str,
                 higher_is_better: bool = True,
                 risk_free_rate: float = 0.0,
                 verbose: bool = True):
        """
        Initializes the GridSearchOptimizer.

        Args:
            backtester_class: The backtester class (e.g., SimpleBacktester).
            fixed_params (dict): Dictionary of parameters that remain constant during the search
                                 (e.g., {'tickers': [...], 'start_date': '...', 'end_date': '...',
                                         'initial_capital': ..., 'commission_per_trade': ...,
                                         'strategy_name': '...'}).
                                 Must include 'strategy_name'.
            param_grid (dict): Dictionary defining the search space. Keys are parameter names
                               (e.g., 'window', 'num_std', 'interval', 'trade_size_fraction'),
                               values are lists of values to test.
            metric_to_optimize (str): Name of the metric in the performance results dict to optimize.
            higher_is_better (bool, optional): True if higher values of the metric are better. Defaults to True.
            risk_free_rate (float, optional): Risk-free rate for performance calculation. Defaults to 0.0.
            verbose (bool, optional): If True, print progress messages during the search. Defaults to True.
        """
        self.backtester_class = backtester_class
        self.fixed_params = fixed_params
        self.param_grid = param_grid
        self.metric_to_optimize = metric_to_optimize
        self.higher_is_better = higher_is_better
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose

        self.results_df = None
        self.best_params = None
        self.best_score = -np.inf if higher_is_better else np.inf

        if 'strategy_name' not in fixed_params:
             raise ValueError("fixed_params must include 'strategy_name'.")

    def _generate_combinations(self):
        """Generates all parameter combinations from the param_grid."""
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run_search(self):
        """Executes the grid search optimization."""
        parameter_combinations = self._generate_combinations()
        num_combinations = len(parameter_combinations)
        results_list = []
        start_time = time.time()

        if self.verbose:
            print(f"Starting Grid Search with {num_combinations} combinations...")
            print(f"Optimizing for: {self.metric_to_optimize} ({'Higher' if self.higher_is_better else 'Lower'} is better)")
            print(f"Fixed Parameters: {self.fixed_params}")
            print(f"Parameter Grid: {self.param_grid}")
            print("-" * 30)


        for i, params in enumerate(parameter_combinations):
            if self.verbose:
                print(f"Running combination {i+1}/{num_combinations}: {params}")

            # --- Prepare parameters for the specific backtester run ---
            current_run_params = self.fixed_params.copy()

            # Strategy parameters need special handling - they might be part of the grid
            strategy_params_in_grid = {k: v for k, v in params.items() if k not in ['interval', 'trade_size_fraction']}
            strategy_params_from_fixed = self.fixed_params.get('strategy_params', {})
            # Combine fixed strategy params with those from the grid (grid values override fixed ones)
            current_run_params['strategy_params'] = {**strategy_params_from_fixed, **strategy_params_in_grid}

            # Add other parameters from the grid that are direct args to the backtester init
            if 'interval' in params:
                current_run_params['interval'] = params['interval']
            if 'trade_size_fraction' in params:
                current_run_params['trade_size_fraction'] = params['trade_size_fraction']

            # Remove grid keys that are now part of strategy_params or direct args
            # to avoid passing them as top-level args if not needed by init
            grid_keys_used_in_strat_params = list(strategy_params_in_grid.keys())
            backtester_direct_args_in_grid = ['interval', 'trade_size_fraction']
            all_grid_keys_used = set(grid_keys_used_in_strat_params + backtester_direct_args_in_grid)


            try:
                # Instantiate the backtester
                backtester = self.backtester_class(**current_run_params)

                # Run the backtest
                backtester.run() # Ideally, modify run() to have optional verbosity

                # Calculate performance
                # Ideally, modify calculate_performance() to have optional verbosity
                performance = backtester.calculate_performance(risk_free_rate=self.risk_free_rate)

                if performance:
                    metric_value = performance.get(self.metric_to_optimize)
                    if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
                         metric_value = -np.inf if self.higher_is_better else np.inf
                         if self.verbose:
                             print(f"  -> Warning: Metric '{self.metric_to_optimize}' not found or NaN.")

                    results_list.append({
                        **params, # Record the grid parameters used
                        **performance # Add all performance metrics
                    })
                    if self.verbose:
                        print(f"  -> Result: {self.metric_to_optimize} = {metric_value:.4f}")

                    # Update best score if necessary
                    is_better = (self.higher_is_better and metric_value > self.best_score) or \
                                (not self.higher_is_better and metric_value < self.best_score)
                    if is_better:
                        self.best_score = metric_value
                        self.best_params = params

                else:
                    if self.verbose:
                        print("  -> No performance data returned.")
                    results_list.append({**params, self.metric_to_optimize: -np.inf if self.higher_is_better else np.inf})

            except Exception as e:
                if self.verbose:
                    print(f"  -> ERROR running combination {params}: {e}")
                # Record failure, assign a poor score
                results_list.append({**params, self.metric_to_optimize: -np.inf if self.higher_is_better else np.inf})

        # --- Process Results ---
        if results_list:
            self.results_df = pd.DataFrame(results_list)
            self.results_df = self.results_df.sort_values(
                by=self.metric_to_optimize,
                ascending=(not self.higher_is_better),
                na_position='last' # Put NaN results at the bottom
            )
            # Ensure best_params reflects the actual top row after sorting NaNs
            if not self.results_df.empty:
                top_metric = self.results_df.iloc[0][self.metric_to_optimize]
                if top_metric is not None and not (isinstance(top_metric, float) and np.isnan(top_metric)):
                    self.best_score = top_metric
                    # Extract only the grid parameters for best_params
                    self.best_params = self.results_df.iloc[0][list(self.param_grid.keys())].to_dict()
                else:
                    # Handle case where all results might be NaN/errors
                    self.best_params = None
                    self.best_score = -np.inf if self.higher_is_better else np.inf


        end_time = time.time()
        if self.verbose:
            print("-" * 30)
            print(f"Grid Search finished in {end_time - start_time:.2f} seconds.")
            if self.best_params:
                print(f"Best {self.metric_to_optimize}: {self.best_score:.4f}")
                print(f"Best Parameters: {self.best_params}")
            else:
                 print("No successful combination found.")
            print("-" * 30)

        return self.results_df, self.best_params

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Grid Search Optimizer Example...")

    # --- Fixed Parameters for the Backtest ---
    fixed_params_example = {
        'tickers': ['AAPL', 'GOOGL'], # Example tickers
        'start_date': '2024-01-01',     # Example date range
        'end_date': '2024-06-30',
        'initial_capital': 10000.0,
        'commission_per_trade': 1.0,
        'strategy_name': 'bollinger_bands' # Must match strategy
    }

    # --- Grid Search Parameter Space ---
    param_grid_example = {
        # Strategy-specific parameters (must match SimpleBacktester expectation for strategy_params)
        'window': [15, 20, 25],
        'num_std': [1.5, 2.0],
        # Backtester direct parameters
        'interval': ['1h', '1d'],
        'trade_size_fraction': [0.15, 0.25]
    }

    # --- Optimization Target ---
    metric = 'Sharpe Ratio'
    higher_better = True

    # --- Instantiate and Run ---
    optimizer = GridSearchOptimizer(
        backtester_class=SimpleBacktester,
        fixed_params=fixed_params_example,
        param_grid=param_grid_example,
        metric_to_optimize=metric,
        higher_is_better=higher_better,
        verbose=True # Show progress
    )

    # Run the search
    results_dataframe, best_parameters = optimizer.run_search()

    # --- Display Results ---
    if results_dataframe is not None:
        print("--- Optimization Results Summary ---")
        # Display top 5 results
        print(results_dataframe.head().to_string())

        print("--- Best Parameters Found ---")
        print(best_parameters)

        print("--- WARNING: Overfitting Risk ---")
        print("Parameters optimized on in-sample data. Validate on out-of-sample data.")
    else:
        print("Optimization did not produce any results.")
