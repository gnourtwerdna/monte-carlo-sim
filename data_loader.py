import yfinance as yf
import pandas as pd
import os

# Define sectors and stock tickers
portfolio = {
    "Technology": ["AAPL", "MSFT", "NVDA", "CRWD", "PANW"],
    "Consumer_Cyclical": ["TSLA", "HD", "MCD", "DPZ", "RCL"],
    "Healthcare": ["JNJ", "PFE", "UNH", "DXCM", "ZBH"],
    "Energy": ["XOM", "CVX", "COP", "ENPH", "FSLR"],
    "Financials": ["JPM", "BAC", "WFC", "SCHW", "GS"]
}

def fetch_stock_data(ticker, start="2000-01-01", end="2024-12-31", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start (str): Start date in "YYYY-MM-DD" format.
        end (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval ("1d", "1wk", "1mo").

    Returns:
        pd.DataFrame: Cleaned stock data.
    """
    try:
        stock = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
        stock.dropna(inplace=True)  
        stock.reset_index(inplace=True)  
        return stock
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    base_folder = "stock_data"  
    os.makedirs(base_folder, exist_ok=True)

    for sector, tickers in portfolio.items():
        sector_folder = os.path.join(base_folder, sector)
        os.makedirs(sector_folder, exist_ok=True)  

        print(f"\nFetching data for {sector} sector...")
        for ticker in tickers:
            stock_df = fetch_stock_data(ticker, start="2000-01-01", end="2024-12-31")
            if stock_df is not None:
                file_path = os.path.join(sector_folder, f"{ticker}.csv")
                stock_df.to_csv(file_path, index=False)
                print(f"✔ {ticker} data saved to {file_path} ({len(stock_df)} rows).")

    print("\nAll stock data has been successfully saved to CSV files.")
