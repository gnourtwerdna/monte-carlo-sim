import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from simulation import run_monte_carlo_portfolio_simulation
import re

portfolio_dict = {
    "Technology": ["AAPL", "MSFT", "NVDA", "CRWD", "PANW"],
    "Consumer_Cyclical": ["TSLA", "HD", "MCD", "DPZ", "RCL"],
    "Healthcare": ["JNJ", "PFE", "UNH", "DXCM", "ZBH"],
    "Energy": ["XOM", "CVX", "COP", "ENPH", "FSLR"],
    "Financials": ["JPM", "BAC", "WFC", "SCHW", "GS"]
}

@st.cache_data
def fetch_live_data(tickers_list_tuple, start_date_str):
    """
    Fetches live daily Adj Close data from yfinance for a list of tickers
    from start_date to the current date. Uses Streamlit caching.

    Args:
        tickers_list_tuple (tuple): Tuple of UPPERCASE ticker symbols.
        start_date_str (str): Start date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame or None: Cleaned DataFrame with Adj Close prices,
                              DatetimeIndex, tickers as columns. Or None on error.
    """
    tickers_list = list(tickers_list_tuple)
    end_date_str = date.today().strftime('%Y-%m-%d')
    st.info(f"Fetching live data for {len(tickers_list)} tickers ({start_date_str} to {end_date_str})...")
    try:
        data = yf.download(
            tickers = tickers_list,
            start = start_date_str,
            end = end_date_str,
            interval = "1d",
            auto_adjust = False,
            progress = False,
            group_by = 'ticker'
        )

        if data.empty:
            st.error("No data returned from yfinance for the selected period.")
            return None

        adj_close_list = []
        successfully_loaded_tickers = []
        for ticker in tickers_list:
            try:
                s = data[ticker]['Adj Close']
                s.name = ticker
                adj_close_list.append(s)
                successfully_loaded_tickers.append(ticker)
            except KeyError:
                st.warning(f"No data found for ticker: {ticker}")
            except Exception as e_inner:
                 st.warning(f"Error processing ticker {ticker}: {e_inner}")

        if not adj_close_list:
             st.error("Could not extract Adj Close for any requested ticker.")
             return None

        adj_close_data = pd.concat(adj_close_list, axis=1)
        cleaned_data = adj_close_data.dropna()

        if cleaned_data.empty:
             st.warning("Dataframe empty after removing rows with missing values (check ticker history overlap).")
             return None

        final_tickers = cleaned_data.columns.tolist()
        st.write(f"Using data for: {', '.join(final_tickers)}")
        return cleaned_data

    except yf.YFRateLimitError as rle:
         st.error(f"Yahoo Finance Rate Limit Error: {rle}. Please wait a while before trying again.")
         return None
    except Exception as e:
        st.error(f"Error during yfinance download or processing: {e}")
        return None

def generate_plots(results_df, initial_investment, title_suffix="", num_paths_to_plot=100):
    """
    Generates interactive Plotly figures for results (simulation paths and distribution).

    Args:
        results_df (pd.DataFrame): DataFrame where each column is a simulated
            portfolio value path and index is the simulated day.
        initial_investment (float): The starting value of the portfolio ($).
        title_suffix (str, optional): Text to append to plot titles. Defaults to "".
        num_paths_to_plot (int, optional): Max number of paths to display on paths plot.
            Defaults to 100.

    Returns:
        tuple: A tuple containing two Plotly Figure objects: `(fig_paths, fig_hist)`
    """
    # Simulation paths plot
    fig_paths = go.Figure(); paths_to_plot = min(results_df.shape[1], num_paths_to_plot); x_values = results_df.index.tolist()
    for i in range(paths_to_plot): fig_paths.add_trace(go.Scatter(x=x_values, y=results_df.iloc[:, i], mode='lines', line=dict(width=0.8), opacity=0.6, showlegend=False))
    fig_paths.add_trace(go.Scatter(x=[x_values[0], x_values[-1]], y=[initial_investment, initial_investment], mode='lines', line=dict(color='red', dash='dash', width=2), name=f'Initial Investment (${initial_investment:,.0f})'))
    fig_paths.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#FFFFFF"), title={'text': f"Simulation Paths {title_suffix}", 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Simulated Trading Days', yaxis_title='Portfolio Value ($)', legend=dict(x=0.01, y=0.99, font=dict(size=10)), margin=dict(l=40, r=40, t=80, b=40))
    fig_paths.update_xaxes(gridcolor='gray'); fig_paths.update_yaxes(gridcolor='gray')


    # Final value distribution plot
    final_values = results_df.iloc[-1]
    mean_val = final_values.mean(); median_val = final_values.median()
    quantile_05 = final_values.quantile(0.05); quantile_95 = final_values.quantile(0.95)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_values, nbinsx=50, histnorm='probability density', marker_line_color="white", marker_line_width=1, opacity=0.8, name="Distribution"))

    # Annotations for lines
    hist_data, bin_edges = np.histogram(final_values, bins=50, density=True)
    max_density = hist_data.max() if len(hist_data) > 0 else 0.0001
    y_level_1 = max_density * 1.02; y_level_2 = max_density * 1.12; y_level_3 = max_density * 1.22
    annot_y_q05 = y_level_1; annot_y_median = y_level_2; annot_y_mean = y_level_3; annot_y_q95 = y_level_1
    annotations = [
        dict(x=mean_val, y=annot_y_mean, xref='x', yref='y', text="Mean", showarrow=False, font=dict(color="red", size=11)),
        dict(x=median_val, y=annot_y_median, xref='x', yref='y', text="Median", showarrow=False, font=dict(color="white", size=11)),
        dict(x=quantile_05, y=annot_y_q05, xref='x', yref='y', text="5% Quantile", showarrow=False, font=dict(color="darkorange", size=11)),
        dict(x=quantile_95, y=annot_y_q95, xref='x', yref='y', text="95% Quantile", showarrow=False, font=dict(color="darkorange", size=11))
    ]
    fig_hist.update_layout(annotations=annotations)

    # Vertical lines
    fig_hist.add_vline(x=mean_val, line_color="red", line_dash="dash", name="Mean")
    fig_hist.add_vline(x=median_val, line_color="white", line_dash="dash", name="Median")
    fig_hist.add_vline(x=quantile_05, line_color="darkorange", line_dash="dot", name="5% Quantile")
    fig_hist.add_vline(x=quantile_95, line_color="darkorange", line_dash="dot", name="95% Quantile")

    # Layout update
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#FFFFFF"), title={'text': f"Final Value Distribution {title_suffix}", 'x': 0.5, 'xanchor': 'center'}, xaxis_title='Final Portfolio Value ($)', yaxis_title='Probability Density', margin=dict(l=40, r=40, t=80, b=40), showlegend=False)
    fig_hist.update_xaxes(gridcolor='gray'); fig_hist.update_yaxes(gridcolor='gray')

    return fig_paths, fig_hist


# Streamlit app
st.set_page_config(
    page_title="Monte Carlo Portfolio Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Monte Carlo Portfolio Simulator")
st.caption("Work of Terry and Andrew")
st.markdown("This tool simulates and visualizes potential portfolio outcomes using historical data and Monte Carlo methods.")

# Sidebar with inputs
with st.sidebar:
    st.header("1. Portfolio Selection")

    input_mode = st.radio(
        "Choose Input Method:",
        ('Select Predefined Sector', 'Enter Custom Ticker List'),
        key="portfolio_mode"
    )

    tickers_to_run = [] 
    portfolio_description = "N/A" 

    if input_mode == 'Select Predefined Sector':
        available_sectors = sorted(list(portfolio_dict.keys()))
        selected_sector = st.selectbox("Select Sector", options=available_sectors, key="sector_select")
        if selected_sector in portfolio_dict:
            tickers_to_run = portfolio_dict[selected_sector]
            portfolio_description = f"{selected_sector} Sector"
        else:
            st.error("Selected sector not found.") 

    elif input_mode == 'Enter Custom Ticker List':
        ticker_input_string = st.text_area(
            "Enter Tickers (comma/space/newline separated):",
            placeholder="e.g., AAPL, MSFT, JNJ, XOM",
            height=100,
            key="ticker_input"
        )
        if ticker_input_string:
            tickers_raw = re.split(r'[,\s\n]+', ticker_input_string)
            custom_tickers_list = sorted(list(set([
                ticker.strip().upper() for ticker in tickers_raw if ticker.strip()
            ])))
            if custom_tickers_list:
                tickers_to_run = custom_tickers_list # Use these tickers
                portfolio_description = f"Custom ({len(tickers_to_run)} tickers)"
                st.write(f"Tickers to fetch: {', '.join(tickers_to_run)}") # Show parsed tickers
            else:
                st.warning("Please enter valid ticker symbols.")
        else:
             st.info("Enter ticker symbols above.")


    st.header("2. Simulation Settings")
    num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)
    num_days = st.number_input("Trading Days (Horizon)", min_value=30, max_value=252*5, value=252, step=21)
    initial_investment = st.number_input("Initial Investment ($)", min_value=1000, max_value=1_000_000, value=10_000, step=1000)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    risk_free_rate_decimal = risk_free_rate / 100.0

    st.markdown("---")
    # Disable button if no valid tickers are ready
    run_button_disabled = len(tickers_to_run) < 2
    run_button = st.button("Run Simulation", disabled=run_button_disabled)
    if run_button_disabled and (input_mode == 'Enter Custom Ticker List' and ticker_input_string):
         st.warning("Enter at least two valid tickers to enable simulation.")


if run_button:
    st.markdown("---")
    start_date = "2000-01-01"

    with st.spinner(f"Fetching LATEST data ({start_date} to Today) for {portfolio_description}..."):
        hist_price_data = fetch_live_data(tuple(sorted(tickers_to_run)), start_date)


    if hist_price_data is not None and not hist_price_data.empty:
        with st.spinner(f"Running {num_simulations} simulations for {portfolio_description}..."):
            tickers = hist_price_data.columns.tolist()
            if len(tickers) < 2:
                    st.error(f"Simulation requires at least 2 assets with overlapping data, only found {len(tickers)}: {', '.join(tickers)}")
            else:
                weights = np.ones(len(tickers)) / len(tickers)

                with st.expander("Portfolio Composition (Based on Fetched Data)", expanded=False):
                    st.write("Assets Simulated:", ", ".join(tickers))
                    st.write("Weights:", f"{weights[0]:.2%} per asset (Equal Weighted).")

                results_df = run_monte_carlo_portfolio_simulation(
                    price_data=hist_price_data, weights=weights, num_simulations=int(num_simulations),
                    num_days=int(num_days), initial_investment=initial_investment
                )

                if results_df is not None and not results_df.empty:
                    st.success("Simulation Complete!")
                    final_vals = results_df.iloc[-1]; quantile_05 = final_vals.quantile(0.05); quantile_95 = final_vals.quantile(0.95); mean_val = final_vals.mean(); median_val = final_vals.median(); cvar_05 = final_vals[final_vals <= quantile_05].mean()
                    delta_val_var = quantile_05 - initial_investment; delta_val_cvar = cvar_05 - initial_investment; delta_val_q95 = quantile_95 - initial_investment
                    start_row = pd.DataFrame({col: [initial_investment] for col in results_df.columns}); results_with_start = pd.concat([start_row, results_df], ignore_index=True)
                    simulated_returns_df = results_with_start.pct_change().dropna(); paths_daily_std = simulated_returns_df.std(axis=0); avg_daily_std = paths_daily_std.mean()
                    annualized_volatility = avg_daily_std * np.sqrt(252)
                    total_return = (final_vals / initial_investment) - 1; avg_annualized_return = ((1 + total_return).mean())**(252.0/num_days) - 1
                    if annualized_volatility > 1e-6: sharpe_ratio = (avg_annualized_return - risk_free_rate_decimal) / annualized_volatility
                    else: sharpe_ratio = np.nan
                    drawdowns = []; worst_mdd = 0
                    for i in range(results_df.shape[1]):
                            path = results_df.iloc[:, i]; path_with_start = pd.concat([pd.Series([initial_investment]), path], ignore_index=True)
                            running_max = path_with_start.cummax(); drawdown = (path_with_start / running_max) - 1
                            max_drawdown_path = drawdown.min(); drawdowns.append(max_drawdown_path)
                            if max_drawdown_path < worst_mdd: worst_mdd = max_drawdown_path
                    avg_mdd = np.mean(drawdowns)

                    st.subheader("Simulation Metrics")
                    perf_tab, risk_tab = st.tabs(["Performance Summary", "Risk Analysis"])
                    with perf_tab:
                            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                            p_col1.metric("Avg. Annual Return", f"{avg_annualized_return:.2%}")
                            p_col2.metric("Mean Final Value", f"${mean_val:,.2f}")
                            p_col3.metric("Median Final Value", f"${median_val:,.2f}")
                            p_col4.metric(label="95% Quantile (Upside)", value=f"${quantile_95:,.2f}", delta=round(delta_val_q95, 2))
                    with risk_tab:
                            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
                            r_col1.metric("Annual Volatility", f"{annualized_volatility:.2%}")
                            r_col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")
                            r_col3.metric("Avg. Max Drawdown", f"{abs(avg_mdd):.2%}")
                            r_col4.metric(label="5% VaR (Value at Risk)", value=f"${quantile_05:,.2f}", delta=round(delta_val_var, 2))
                            r_col5.metric(label="5% CVaR (Expected Shortfall)", value=f"${cvar_05:,.2f}", delta=round(delta_val_cvar, 2))

                    st.markdown("---")
                    st.subheader("Simulation Visualizations")
                    fig_paths, fig_hist = generate_plots(results_df, initial_investment, f"({portfolio_description})")
                    if fig_paths and fig_hist:
                        plot_tab1, plot_tab2 = st.tabs(["Simulation Paths", "Final Distribution"])
                        with plot_tab1: st.plotly_chart(fig_paths, use_container_width=True)
                        with plot_tab2: st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.warning("Unable to render charts.")
                else:
                    st.error("Simulation did not return valid data.")
    elif run_button:
         st.error(f"Failed to load necessary data before running simulation.")

else:
    st.info("Configure portfolio and simulation settings in the sidebar, then click 'Run Simulation' to start.")

