import pandas as pd  # pandas for DataFrame manipulation
import numpy as np   # numpy for numerical calculations


def add_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA) — the average closing price over the last N days.

    This smooths out day-to-day noise so you can see the overall trend.
    e.g. a 20-day SMA averages the last 20 closing prices into one value per day.

    Args:
        df:     OHLCV DataFrame from data.py
        window: how many days to average over, e.g. 20 for a 20-day SMA

    Returns:
        a pandas Series (single column of values) with the SMA for each date
    """
    # .rolling(window) creates a sliding window of N rows
    # .mean() computes the average within that window
    # the first (window - 1) rows will be NaN because there aren't enough prior days yet
    return df["Close"].rolling(window=window).mean()


def add_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) — similar to SMA but gives more weight to recent prices.

    Unlike SMA which treats all days equally, EMA reacts faster to recent price changes.
    Traders often use this to spot momentum shifts earlier than SMA would.

    Args:
        df:     OHLCV DataFrame
        window: the "span" — higher = smoother but slower to react

    Returns:
        a pandas Series with the EMA for each date
    """
    # .ewm() = exponential weighted moving — span= controls how much weight recent values get
    # adjust=False uses the standard recursive EMA formula traders actually use
    return df["Close"].ewm(span=window, adjust=False).mean()


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — a momentum indicator that measures whether a stock
    is overbought (too high, likely to drop) or oversold (too low, likely to rise).

    RSI ranges from 0 to 100:
        above 70 = overbought (potential sell signal)
        below 30 = oversold (potential buy signal)
        50       = neutral

    The standard window is 14 days, which is what most traders use.

    Args:
        df:     OHLCV DataFrame
        window: lookback period, default 14 days

    Returns:
        a pandas Series with RSI values between 0 and 100
    """
    # calculate how much the price changed each day (positive = up, negative = down)
    delta = df["Close"].diff()  # .diff() subtracts each row from the previous one

    # separate the daily changes into gains (positive days) and losses (negative days)
    # .clip() clamps values — clip(lower=0) sets all negatives to 0, clip(upper=0) sets all positives to 0
    gain = delta.clip(lower=0)   # only keep positive price changes, set negatives to 0
    loss = -delta.clip(upper=0)  # only keep negative changes, flip sign so losses are positive numbers

    # compute the average gain and average loss over the window using EMA smoothing
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    # RS = ratio of average gain to average loss
    # if avg_loss is 0 (stock never dropped), RS would be infinite — pandas handles this as NaN
    rs = avg_gain / avg_loss

    # convert RS into the 0-100 RSI scale using the standard formula
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Bollinger Bands — a volatility indicator consisting of three lines:
        Upper Band: SMA + 2 standard deviations  (price is "high" relative to recent range)
        Middle Band: SMA (the 20-day moving average)
        Lower Band: SMA - 2 standard deviations  (price is "low" relative to recent range)

    When price touches the upper band, the stock may be overbought.
    When price touches the lower band, the stock may be oversold.
    When the bands are wide = high volatility. Narrow = low volatility.

    Args:
        df:     OHLCV DataFrame
        window: lookback period, default 20 days

    Returns:
        a tuple of three pandas Series: (upper_band, middle_band, lower_band)
    """
    # middle band is just the standard SMA
    middle = df["Close"].rolling(window=window).mean()

    # standard deviation measures how spread out prices are within the window
    # higher std = more volatile = wider bands
    std = df["Close"].rolling(window=window).std()

    # upper and lower bands are 2 standard deviations above and below the middle
    # the "2" is the standard multiplier used in finance — covers ~95% of price action
    upper = middle + (2 * std)
    lower = middle - (2 * std)

    return upper, middle, lower


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering functions and adds results as new columns to the DataFrame.

    This is the main function we'll call from other files — it takes a clean OHLCV DataFrame
    and returns it with all technical indicators added as extra columns.

    Args:
        df: OHLCV DataFrame from data.py

    Returns:
        the same DataFrame with new columns: SMA_20, SMA_50, EMA_20, RSI, BB_Upper, BB_Middle, BB_Lower
    """
    result = df.copy()  # always work on a copy so we don't modify the original

    # add simple moving averages for two common timeframes
    result["SMA_20"] = add_sma(df, window=20)   # short-term trend (1 month)
    result["SMA_50"] = add_sma(df, window=50)   # medium-term trend (2.5 months)

    # add exponential moving average
    result["EMA_20"] = add_ema(df, window=20)

    # add RSI momentum indicator
    result["RSI"] = add_rsi(df)

    # add bollinger bands — unpack the tuple directly into three columns
    result["BB_Upper"], result["BB_Middle"], result["BB_Lower"] = add_bollinger_bands(df)

    return result