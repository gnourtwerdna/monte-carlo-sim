import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from simulation import load_data_from_folder, run_monte_carlo_portfolio_simulation

@st.cache_data
def cached_load_data(folder):
    """
    Loads and caches sector data from a specified folder using st.cache_data.

    Args:
        folder (str): Path to the folder containing stock CSV files for the sector.

    Returns:
        pd.DataFrame or None: DataFrame with price data loaded by
            load_data_from_folder, or None if loading failed.
    """
    st.write(f"Loading data for: {os.path.basename(folder)}")
    return load_data_from_folder(folder)

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
    st.header("Simulation Settings")
    base_data_folder = "stock_data"
    available_sectors = sorted([d for d in os.listdir(base_data_folder) if os.path.isdir(os.path.join(base_data_folder, d))])
    selected_sector = st.selectbox("Select Sector", options=available_sectors)
    target_folder = os.path.join(base_data_folder, selected_sector)
    st.markdown("---")
    num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
    num_days = st.number_input("Trading Days (Horizon)", min_value=30, max_value=252*5, value=252, step=21)
    initial_investment = st.number_input("Initial Investment ($)", min_value=1000, max_value=1_000_000, value=10_000, step=1000)
    st.markdown("---")
    risk_free_rate = st.number_input("Risk-Free Rate (%) for Sharpe Ratio:", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    risk_free_rate_decimal = risk_free_rate / 100.0
    st.markdown("---")
    run_button = st.button("Run Simulation")

# Main panel
if run_button:
    with st.spinner(f"Running {num_simulations} simulations..."):
        hist_price_data = cached_load_data(target_folder)

        if hist_price_data is None or hist_price_data.empty:
            st.error("Historical data could not be loaded.")
        else:
            tickers = hist_price_data.columns.tolist()
            weights = np.ones(len(tickers)) / len(tickers)

            with st.expander("Portfolio Composition", expanded=False):
                st.write("Assets:", ", ".join(tickers))
                st.write("Weights:", f"{weights[0]:.2%} per asset (Equal Weighted).")

            results_df = run_monte_carlo_portfolio_simulation(
                price_data=hist_price_data, weights=weights, num_simulations=int(num_simulations),
                num_days=int(num_days), initial_investment=initial_investment
            )

            if results_df is not None and not results_df.empty:
                st.success("Simulation Completed Successfully!")
                st.markdown("---")

                # Calculate metrics
                final_vals = results_df.iloc[-1]
                quantile_05 = final_vals.quantile(0.05)
                quantile_95 = final_vals.quantile(0.95)
                mean_val = final_vals.mean() # Mean final value
                median_val = final_vals.median() # Median final value
                cvar_05 = final_vals[final_vals <= quantile_05].mean()
                start_row = pd.DataFrame({col: [initial_investment] for col in results_df.columns})
                results_with_start = pd.concat([start_row, results_df], ignore_index=True)
                simulated_returns_df = results_with_start.pct_change().dropna()
                paths_daily_std = simulated_returns_df.std(axis=0)
                avg_daily_std = paths_daily_std.mean()
                annualized_volatility = avg_daily_std * np.sqrt(252)
                total_return = (final_vals / initial_investment) - 1
                avg_annualized_return = ((1 + total_return).mean())**(252.0/num_days) - 1
                if annualized_volatility > 1e-6: sharpe_ratio = (avg_annualized_return - risk_free_rate_decimal) / annualized_volatility
                else: sharpe_ratio = np.nan
                drawdowns = []; worst_mdd = 0
                for i in range(results_df.shape[1]):
                    path = results_df.iloc[:, i]; path_with_start = pd.concat([pd.Series([initial_investment]), path], ignore_index=True)
                    running_max = path_with_start.cummax(); drawdown = (path_with_start / running_max) - 1
                    max_drawdown_path = drawdown.min(); drawdowns.append(max_drawdown_path)
                    if max_drawdown_path < worst_mdd: worst_mdd = max_drawdown_path
                avg_mdd = np.mean(drawdowns)

                # Calculate numeric deltas
                delta_val_var = quantile_05 - initial_investment
                delta_val_cvar = cvar_05 - initial_investment
                delta_val_q95 = quantile_95 - initial_investment

                # Display performance and risk metrics using tabs
                st.subheader("Simulation Metrics")
                perf_tab, risk_tab = st.tabs(["Performance Summary", "Risk Analysis"])

                # Performance tab
                with perf_tab:
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Avg. Annual Return", f"{avg_annualized_return:.2%}")
                    p_col2.metric("Mean Final Value", f"${mean_val:,.2f}")
                    p_col3.metric("Median Final Value", f"${median_val:,.2f}")
                    p_col4.metric(label="95% Quantile (Upside)", value=f"${quantile_95:,.2f}", delta=round(delta_val_q95, 2))

                # Risk tab
                with risk_tab:
                    r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
                    r_col1.metric("Annual Volatility", f"{annualized_volatility:.2%}")
                    r_col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")
                    r_col3.metric("Avg. Max Drawdown", f"{abs(avg_mdd):.2%}")
                    r_col4.metric(label="5% VaR (Value at Risk)", value=f"${quantile_05:,.2f}", delta=round(delta_val_var, 2))
                    r_col5.metric(label="5% CVaR (Expected Shortfall)", value=f"${cvar_05:,.2f}", delta=round(delta_val_cvar, 2))

                st.markdown("---")
                st.subheader("Simulation Visualizations")
                fig_paths, fig_hist = generate_plots(results_df, initial_investment, f"({selected_sector})") 
                if fig_paths and fig_hist:
                    tab1, tab2 = st.tabs(["Simulation Paths", "Final Distribution"])
                    with tab1: st.plotly_chart(fig_paths, use_container_width=True)
                    with tab2: st.plotly_chart(fig_hist, use_container_width=True)
                else: st.warning("Unable to render charts.")
            else: st.error("Simulation did not return valid data.")
else: st.info("Adjust parameters and click 'Run Simulation' to begin.")