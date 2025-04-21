# Monte Carlo Simulation for Multi-Sector Portfolio Risk and Return Analysis

**Authors:** Andrew Truong, Terry Tu
**(ISYE 6644 Project)**

## Description

This project implements a Monte Carlo simulation engine in Python to analyze and visualize the potential risk and return profiles of multi-sector stock portfolios. It calculates key performance and risk metrics over different time periods (1, 3, and 5 years) and presents the results through an interactive web dashboard built with Streamlit and Plotly.

## Setup
Install the required Python packages listed in `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Collection (Run if `stock_data` is empty or needs update)

* Run the data loader script:
    ```bash
    python data_loader.py
    ```

### 2. Running the Dashboard

Once the data is available in the `stock_data` folder:

* Run the Streamlit application:
    ```bash
    streamlit run dashboard.py
    ```

## Dashboard

* Select the desired **Sector** portfolio via a dropdown menu.
* Configure simulation parameters using sidebar inputs:
    * **Number of Simulations**
    * **Simulation Horizon** (Trading Days)
    * **Initial Investment** ($)
    * **Risk-Free Rate** (%) for Sharpe Ratio
* Trigger the simulation with the **"Run Simulation" button**.
* View results presented in interactive charts and metric tabs.