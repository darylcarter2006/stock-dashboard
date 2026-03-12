import yfinance as yf # library to pull stock data from yahoo finance
import pandas as pd

# :str is a type hint to tell what each argument expects
# str means it should be a string, "1y" is the default if one is not passed through
# -> is a return type hint, saying that this function will return a pandas DataFrame
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker from Yahoo Finance.

    OHLCV stands for:
        Open   - price the stock opened at for that day
        High   - highest price reached during that day
        Low    - lowest price reached during that day
        Close  - price the stock closed at (most commonly used)
        Volume - how many shares were traded that day

    Args:
        ticker:   stock symbol, e.g. 'AAPL' for Apple, 'TSLA' for Tesla
        period:   how far back to pull data. options: '1mo', '3mo', '6mo', '1y', '2y', '5y'
        interval: how often to sample. '1d' = one row per day, '1wk' = one row per week

    Returns:
        a pandas DataFrame — basically a table — with dates as rows and OHLCV as columns
    """

    # yf.download makes a request to the yahoo finance library 
    # ticker is the is the stock sybol we are passing in
    # period = period passes the period argument straigt to yfinance
    # auto_adjust = True means yFinance atuomatically adjusts historical prices so the data is consistent
    # progress = False hides the progress bar that yfinance prints by default
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for this ticker: {ticker}")
    
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    df.dropna(inplace=True)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    return df

def get_multiple_tickers(tickers: list, period: str = "1y") -> dict:
    """
    Fetch data for a list of tickers all at once.

    Args:
        tickers: a Python list of ticker strings, e.g. ['AAPL', 'TSLA', 'NVDA']
        period:  how far back to pull for all tickers

    Returns:
        a dictionary where each key is a ticker string and each value is its DataFrame
        e.g. { 'AAPL': <DataFrame>, 'TSLA': <DataFrame> }
    """
    return {ticker: fetch_stock_data(ticker, period=period) for ticker in tickers}