import streamlit as st                          # streamlit turns python scripts into web apps
import pandas as pd                             # DataFrame manipulation
from src.data import fetch_stock_data           # our data fetching module
from src.features import add_all_features       # our feature engineering module
from src.anomaly import detect_anomalies        # our anomaly detection module
from src.charts import build_candlestick_chart  # our chart building module
from src.charts import build_volume_chart       # our volume chart module


# --- PAGE CONFIG ---
# this must be the first streamlit call in the file
# sets the browser tab title, icon, and layout
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide"   # wide layout uses the full browser width instead of a narrow centered column
)


# --- HEADER ---
st.title("📈 Stock Market Analytics Dashboard")
st.markdown("Interactive dashboard for technical analysis and anomaly detection across equities.")


# --- SIDEBAR ---
# st.sidebar puts UI elements in the left panel instead of the main page
# this is where users configure what they want to see
st.sidebar.header("Configuration")

# text input for ticker symbol
# value= sets the default value that appears when the app loads
ticker_input = st.sidebar.text_input(
    label="Ticker Symbol",
    value="AAPL",
    help="Enter a valid US stock ticker e.g. AAPL, TSLA, NVDA, MSFT"  # tooltip on hover
).upper()  # .upper() forces the input to uppercase so 'aapl' becomes 'AAPL'

# dropdown for time period selection
# the first argument is the label, options= is the list of choices
period_options = {
    "3 Months":  "3mo",
    "6 Months":  "6mo",
    "1 Year":    "1y",
    "2 Years":   "2y",
    "5 Years":   "5y"
}

# st.selectbox renders a dropdown — format_func maps the display label to the actual value
selected_period_label = st.sidebar.selectbox(
    label="Time Period",
    options=list(period_options.keys()),  # display the human-readable labels
    index=2                               # default to index 2 = "1 Year"
)
selected_period = period_options[selected_period_label]  # get the actual yfinance period string

# slider for contamination parameter
# this lets the user control how sensitive the anomaly detection is
# min_value, max_value, value (default), step control the slider range
contamination = st.sidebar.slider(
    label="Anomaly Sensitivity",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    help="Higher = more anomalies flagged. Lower = only the most extreme events flagged."
)

# checkboxes for toggling which indicators are shown
st.sidebar.subheader("Indicators")
show_sma    = st.sidebar.checkbox("Show SMA 20 / SMA 50", value=True)
show_bb     = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_volume = st.sidebar.checkbox("Show Volume Chart",    value=True)


# --- DATA LOADING ---
# st.cache_data caches the result of this function so it doesn't re-fetch
# every time the user interacts with the app
# it only re-runs if the arguments (ticker, period) actually change
@st.cache_data
def load_data(ticker: str, period: str, contamination: float) -> pd.DataFrame:
    """
    Fetches data, computes features, and runs anomaly detection.
    Wrapped in cache so it only re-runs when inputs change.
    """
    df = fetch_stock_data(ticker, period=period)
    df = add_all_features(df)
    df = detect_anomalies(df, contamination=contamination)
    return df


# --- MAIN APP LOGIC ---
# st.spinner shows a loading message while the data is being fetched
with st.spinner(f"Loading data for {ticker_input}..."):
    try:
        df = load_data(ticker_input, selected_period, contamination)
    except ValueError as e:
        # if fetch_stock_data raises a ValueError (bad ticker), show an error and stop
        st.error(f"Could not load data: {e}")
        st.stop()  # st.stop() halts execution of the rest of the script


# --- METRICS ROW ---
# st.columns splits the page into side by side columns
# these are summary stat cards at the top of the dashboard
col1, col2, col3, col4, col5 = st.columns(5)

# get the most recent row of data for the summary stats
latest       = df.iloc[-1]   # iloc[-1] = last row
previous     = df.iloc[-2]   # iloc[-2] = second to last row
price_change = latest["Close"] - previous["Close"]  # daily price change in dollars
price_pct    = (price_change / previous["Close"]) * 100  # daily change as a percentage

# st.metric renders a nice card with a label, value, and delta (change indicator)
# delta automatically colors green for positive, red for negative
col1.metric(
    label="Close Price",
    value=f"${latest['Close']:.2f}",          # :.2f formats to 2 decimal places
    delta=f"{price_change:+.2f} ({price_pct:+.1f}%)"  # :+ forces the sign to always show
)
col2.metric(
    label="RSI",
    value=f"{latest['RSI']:.1f}",
    delta="Overbought" if latest["RSI"] > 70 else ("Oversold" if latest["RSI"] < 30 else "Neutral")
)
col3.metric(
    label="Volume",
    value=f"{int(latest['Volume']):,}"  # :, adds comma formatting e.g. 41,120,000
)
col4.metric(
    label="52W High",
    value=f"${df['High'].max():.2f}"   # max of the High column over the selected period
)
col5.metric(
    label="Anomalies Detected",
    value=int(df["is_anomaly"].sum())  # .sum() on a boolean column counts the True values
)


# --- DIVIDER ---
st.divider()  # draws a horizontal line across the page


# --- MAIN CHART ---
st.subheader(f"{ticker_input} Price Action")

# build the chart — pass show_sma and show_bb flags so the chart function can toggle layers
# we need to update build_candlestick_chart to accept these flags
fig = build_candlestick_chart(df, ticker_input, show_sma=show_sma, show_bb=show_bb)
st.plotly_chart(fig, use_container_width=True)  # use_container_width=True makes it fill the page width


# --- VOLUME CHART ---
if show_volume:
    st.subheader("Volume")
    vol_fig = build_volume_chart(df, ticker_input)
    st.plotly_chart(vol_fig, use_container_width=True)


# --- ANOMALY TABLE ---
st.subheader("Flagged Anomalies")

# filter to only anomaly rows and select the most relevant columns to display
anomaly_df = df[df["is_anomaly"] == True][
    ["Close", "Volume", "RSI", "anomaly_score"]
].copy()

# round the float columns for cleaner display
anomaly_df["Close"]         = anomaly_df["Close"].round(2)
anomaly_df["RSI"]           = anomaly_df["RSI"].round(2)
anomaly_df["anomaly_score"] = anomaly_df["anomaly_score"].round(4)

# sort by anomaly score ascending — most anomalous days first
anomaly_df = anomaly_df.sort_values("anomaly_score")

# st.dataframe renders an interactive sortable table
st.dataframe(anomaly_df, use_container_width=True)


# --- FOOTER ---
st.divider()
st.caption("Data sourced from Yahoo Finance via yfinance. For educational purposes only.")