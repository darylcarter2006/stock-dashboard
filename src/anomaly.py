import pandas as pd                                    # DataFrame manipulation
import numpy as np                                     # numerical operations
from sklearn.ensemble import IsolationForest           # the anomaly detection algorithm we're using


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Uses Isolation Forest to detect anomalies in price and volume behavior.

    Isolation Forest works by randomly partitioning the data into a tree structure.
    Normal data points require more partitions to isolate (they blend in with others).
    Anomalies require fewer partitions because they're unusual and stand out.
    The algorithm assigns each row a score — low scores = anomalies.

    contamination controls what percentage of the data we expect to be anomalies.
    0.05 means we're telling the model to flag the most anomalous 5% of trading days.

    Args:
        df:            DataFrame output from add_all_features() in features.py
        contamination: float between 0 and 0.5, percentage of expected anomalies

    Returns:
        the same DataFrame with two new columns added:
            anomaly_score: continuous score, more negative = more anomalous
            is_anomaly:    boolean True/False, True means flagged as anomaly
    """

    result = df.copy()  # always work on a copy, never modify the original

    # these are the features we'll feed into the model
    # we're using Close price, Volume, RSI, and the Bollinger Band width (a volatility measure)
    # dropna() is needed because SMA/RSI have NaN in the first N rows (not enough prior data yet)
    features = result[["Close", "Volume", "RSI", "BB_Upper", "BB_Lower"]].dropna()

    # BB_Width = difference between upper and lower band
    # wider bands = higher volatility — this is a useful signal for the model
    features = features.copy()  # copy again after dropna to avoid pandas SettingWithCopyWarning
    features["BB_Width"] = features["BB_Upper"] - features["BB_Lower"]

    # drop the raw band columns now that we have BB_Width — cleaner input for the model
    features = features.drop(columns=["BB_Upper", "BB_Lower"])

    # initialize the Isolation Forest model
    # n_estimators=100 means it builds 100 trees and averages their results (more = more stable)
    # contamination tells the model what fraction of points to treat as anomalies
    # random_state=42 makes results reproducible — same data will always give same output
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)

    # .fit_predict() does two things in one call:
    #   fit()     — trains the model on our feature data
    #   predict() — immediately scores every row
    # returns an array of 1s and -1s:
    #   1  = normal trading day
    #  -1  = anomaly (unusual price/volume behavior)
    predictions = model.fit_predict(features)

    # .decision_function() returns a continuous anomaly score for each row
    # more negative = more anomalous (further from normal behavior)
    # this gives us nuance beyond just the binary 1/-1 flag
    scores = model.decision_function(features)

    # add the scores and predictions back to the result DataFrame
    # we use features.index instead of result.index because features had rows dropped (dropna)
    # so we need to align on the same dates that actually have valid feature data
    result.loc[features.index, "anomaly_score"] = scores
    result.loc[features.index, "is_anomaly"] = predictions == -1

    return result