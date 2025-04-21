import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

def load_data_from_folder(folder_path):
    """
    Loads, cleans, and combines stock price data from CSV files within a folder.

    Args:
        folder_path (str): Path to the folder containing stock CSVs for one portfolio/sector.

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex, numeric price columns (Adj Close or Close) for each ticker.
    """
    all_data = {}
    loaded_tickers = []
    folder_path = os.path.abspath(folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            ticker = filename[:-4].upper()

            # Read CSV with Date index
            df_ticker = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            price_col_name = 'Adj Close'
            all_data[ticker] = df_ticker[price_col_name]
            loaded_tickers.append(ticker)

    # Cleaning
    portfolio_df = pd.DataFrame(all_data)
    portfolio_df = portfolio_df.reindex(sorted(portfolio_df.columns), axis=1)
    for col in portfolio_df.columns:
        portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors='coerce')
    portfolio_df = portfolio_df.dropna()
    
    return portfolio_df

def run_monte_carlo_portfolio_simulation(price_data, weights, num_simulations, num_days, initial_investment=10000):
    """
    Runs a Monte Carlo simulation to project portfolio value paths based on historical data.

    Args:
        price_data (pd.DataFrame): DataFrame of historical adjusted close prices.
        weights (np.array): Array of portfolio weights for assets.
        num_simulations (int): The number of simulation paths to generate.
        num_days (int): The number of future trading days to simulate.
        initial_investment (float, optional): Starting portfolio value in dollars. Defaults to 10000.

    Returns:
        pd.DataFrame or None: A DataFrame where each column represents a simulated portfolio value path over `num_days`.
    """
    # Check if weights are normalized
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    # Calculate Log Returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()

    # Calculate Mean Returns Vector and Covariance Matrix
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov()

    # Get Last Known Prices
    last_prices = price_data.iloc[-1].values

    # Initialize arrays
    simulated_asset_prices = np.zeros((num_days, num_simulations, len(weights)))
    simulated_portfolio_values = np.zeros((num_days, num_simulations))

    # Run Simulation Loop
    for i in range(num_simulations):
        current_prices = last_prices.copy()
        initial_asset_values = initial_investment * weights
        initial_shares = np.divide(initial_asset_values, last_prices, out=np.zeros_like(initial_asset_values), where=last_prices!=0)
        current_portfolio_value = initial_investment

        for d in range(num_days):
            daily_sim_log_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            current_prices = current_prices * np.exp(daily_sim_log_returns)
            simulated_asset_prices[d, i, :] = current_prices
            # Calculate portfolio value based on initial shares and current prices
            current_portfolio_value = np.sum(initial_shares * current_prices)
            simulated_portfolio_values[d, i] = current_portfolio_value

    portfolio_results_df = pd.DataFrame(simulated_portfolio_values)
    portfolio_results_df.columns = [f'Sim_{j+1}' for j in range(num_simulations)]
    portfolio_results_df.index.name = 'Simulated_Day'

    return portfolio_results_df

def plot_simulation_results(results_df, initial_investment, title_suffix="", num_paths_to_plot=100):
    """
    Generates plots for Monte Carlo simulation results.

    Args:
        results_df (pd.DataFrame): DataFrame of simulation paths (output from simulation function).
        initial_investment (float): The initial investment amount for reference.
        title_suffix (str): Optional suffix for plot titles (e.g., sector name).
        num_paths_to_plot (int): How many simulation paths to display on the line plot.
    """
    num_simulations = results_df.shape[1]
    num_days = results_df.shape[0]
    final_values = results_df.iloc[-1] # Portfolio values on the last simulated day

    # Plot 1: Simulation Paths
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    plt.figure(figsize=(12, 7))

    # Plot a subset of paths to avoid visual clutter and improve performance
    paths_to_plot = min(num_simulations, num_paths_to_plot)
    plt.plot(results_df.index, results_df.iloc[:, :paths_to_plot], lw=0.7, alpha=0.6) # Use results_df index for x-axis

    # Highlight initial investment
    plt.axhline(y=initial_investment, color='red', linestyle='--', linewidth=1.5, label=f'Initial Investment (${initial_investment:,.0f})')

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel(f'Simulated Trading Days')
    plt.title(f'Monte Carlo Portfolio Simulation Paths {title_suffix}\n({paths_to_plot} of {num_simulations} Shown over {num_days} days)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

    # Plot 2: Distribution of Final Values
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50, edgecolor='black', alpha=0.8, density=True) # Use density=True if comparing distributions

    # Add kde line
    final_values.plot(kind='kde', color='blue', lw=1.5, ax=plt.gca(), label='Density Estimate (KDE)')

    plt.xlabel(f'Final Portfolio Value ($) after {num_days} days')
    plt.ylabel('Probability Density') # Or 'Frequency' if density=False
    plt.title(f'Distribution of Final Portfolio Values {title_suffix}')

    # Add lines for mean, median, quantiles
    mean_val = final_values.mean()
    median_val = final_values.median()
    q5 = final_values.quantile(0.05)
    q95 = final_values.quantile(0.95)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: ${mean_val:,.2f}')
    plt.axvline(median_val, color='black', linestyle='dashed', linewidth=1.5, label=f'Median: ${median_val:,.2f}')
    plt.axvline(q5, color='darkorange', linestyle='dotted', linewidth=1.5, label=f'5% Quantile: ${q5:,.2f}')
    plt.axvline(q95, color='darkorange', linestyle='dotted', linewidth=1.5, label=f'95% Quantile: ${q95:,.2f}')

    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    target_folder = "stock_data/Consumer_Cyclical"

    # Simulation parameters
    NUM_SIMULATIONS = 1000
    NUM_DAYS = 252
    INITIAL_INVESTMENT = 10000

    # Load data
    hist_price_data = load_data_from_folder(target_folder)

    # Run simulation
    if hist_price_data is not None and not hist_price_data.empty:
        loaded_tickers = hist_price_data.columns.tolist()
        num_assets = len(loaded_tickers)
        portfolio_weights = np.ones(num_assets) / num_assets

        simulation_results = run_monte_carlo_portfolio_simulation(
            price_data=hist_price_data,
            weights=portfolio_weights,
            num_simulations=NUM_SIMULATIONS,
            num_days=NUM_DAYS,
            initial_investment=INITIAL_INVESTMENT
        )

        # Basic output analysis and plotting
        if simulation_results is not None:
            print("\Simulation Results Summary:")
            final_values = simulation_results.iloc[-1]
            print(f"Initial Investment: ${INITIAL_INVESTMENT:,.2f}")
            print(f"Mean Final Value:   ${final_values.mean():,.2f}")
            print(f"Median Final Value: ${final_values.median():,.2f}")
            print(f"Std Dev Final Value:${final_values.std():,.2f}")
            print(f"\nQuantiles of Final Portfolio Value:")
            print(final_values.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        
            # Extract folder name for title
            folder_name = os.path.basename(target_folder)
            plot_simulation_results(simulation_results, INITIAL_INVESTMENT, title_suffix=f"({folder_name})")