import pandas as pd          # DataFrame manipulation
import plotly.graph_objects as go  # plotly's lower-level API — gives us full control over charts
from plotly.subplots import make_subplots  # lets us stack multiple charts in one figure


def build_candlestick_chart(df: pd.DataFrame, ticker: str, show_sma: bool = True, show_bb: bool = True) -> go.Figure:
    """
    Builds the main price chart with:
        - Candlestick chart (OHLC price action)
        - Bollinger Bands overlay
        - SMA 20 and SMA 50 overlays
        - Anomaly markers (red dots on flagged days)

    A candlestick shows four prices for each day in one visual:
        - The "body" goes from Open to Close
          Green body = price went up that day (Close > Open)
          Red body   = price went down that day (Close < Open)
        - The "wicks" extend to the High and Low of the day

    Args:
        df:     DataFrame output from detect_anomalies() — has all features and anomaly flags
        ticker: stock symbol string, used for chart title

    Returns:
        a Plotly Figure object ready to be rendered in Streamlit
    """

    # make_subplots creates a figure with multiple stacked panels
    # we want 2 rows: top = price chart, bottom = RSI chart
    # row_heights controls the proportional height of each panel
    # shared_xaxes=True links the x-axis so zooming on one panel zooms both
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # price chart gets 70% of height, RSI gets 30%
        vertical_spacing=0.05    # small gap between the two panels
    )

    # --- CANDLESTICK ---
    # go.Candlestick() creates the candlestick chart
    # x = the dates (index of our DataFrame)
    # open, high, low, close = the four OHLCV price columns
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="lime",    # green candles for up days
            decreasing_line_color="red"      # red candles for down days
        ),
        row=1, col=1  # place this in the top panel
    )

    # --- BOLLINGER BANDS ---
    # we add the upper and lower bands as lines, then fill between them
    # this shading shows the "normal" price range visually

    if show_bb:
        # upper band line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.8)", width=1),
                showlegend=True
            ),
            row=1, col=1
        )

        # lower band line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.8)", width=1),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.1)",
                showlegend=True
            ),
            row=1, col=1
        )

    # --- MOVING AVERAGES ---
    if show_sma:
        # SMA 20 — short term trend line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                line=dict(color="orange", width=1.5)
            ),
            row=1, col=1
        )

        # SMA 50 — medium term trend line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line=dict(color="purple", width=1.5)
            ),
            row=1, col=1
        )

    # --- ANOMALY MARKERS ---
    # filter down to only rows where is_anomaly is True
    # these get plotted as red dots on top of the candlestick chart
    anomalies = df[df["is_anomaly"] == True]

    fig.add_trace(
        go.Scatter(
            x=anomalies.index,
            y=anomalies["High"] * 1.01,  # place dot slightly above the candle's high so it's visible
            mode="markers",              # markers = dots only, no connecting lines
            marker=dict(color="red", size=8, symbol="circle"),
            name="Anomaly"
        ),
        row=1, col=1
    )

    # --- RSI PANEL ---
    # RSI goes in the bottom panel (row=2)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color="cyan", width=1.5)
        ),
        row=2, col=1
    )

    # add horizontal reference lines at 70 (overbought) and 30 (oversold)
    # add_hline() draws a horizontal line across the full width of a panel
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # --- LAYOUT ---
    # update_layout controls the overall appearance of the figure
    fig.update_layout(
        title=f"{ticker} — Price Action, Indicators & Anomalies",
        template="plotly_dark",       # dark background theme
        height=700,                   # total figure height in pixels
        xaxis_rangeslider_visible=False,  # hide the range slider under the candlestick (cleaner look)
        legend=dict(
            orientation="h",          # horizontal legend
            yanchor="bottom",
            y=1.02,                   # place legend above the chart
            xanchor="right",
            x=1
        )
    )

    # label the y-axes
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig


def build_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Builds a separate bar chart showing daily trading volume.
    Bars are colored green or red based on whether the stock went up or down that day.

    Args:
        df:     DataFrame with OHLCV and anomaly data
        ticker: stock symbol for the chart title

    Returns:
        a Plotly Figure object
    """

    # determine bar color based on price direction
    # if Close >= Open, it was an up day (green), otherwise down day (red)
    # this uses a list comprehension — compact way to build a list with a condition
    colors = [
        "lime" if close >= open_ else "red"
        for close, open_ in zip(df["Close"], df["Open"])
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,    # apply our green/red color list
            name="Volume"
        )
    )

    fig.update_layout(
        title=f"{ticker} — Daily Volume",
        template="plotly_dark",
        height=300,
        yaxis_title="Volume"
    )

    return fig