# 📈 Stock Market Analytics Dashboard

An interactive financial analytics dashboard for technical analysis and anomaly detection across US equities. Built with Python and deployed as a live web application via Streamlit.

**[Live Demo →](https://darylcarter2.streamlit.app)**

---

## Overview

This dashboard pulls historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance and runs a full analytics pipeline — computing technical indicators, detecting statistically anomalous trading days using machine learning, and rendering everything as an interactive multi-panel chart.

The project is structured with a modular codebase separating data ingestion, feature engineering, anomaly detection, and visualization into independent layers.

---

## Features

- **Candlestick chart** with real-time OHLCV data for any US ticker
- **Technical indicators** — SMA 20, SMA 50, EMA 20, RSI, Bollinger Bands
- **Anomaly detection** via Isolation Forest (scikit-learn) on price, volume, and volatility features
- **Interactive sidebar** — configure ticker, time period, anomaly sensitivity, and indicator toggles
- **Summary metrics** — live close price, RSI status, volume, 52-week high, anomaly count
- **Anomaly table** — sortable table of flagged trading days with scores
- **Volume chart** — daily volume bars colored by price direction

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | yfinance, pandas, NumPy |
| Feature Engineering | pandas, NumPy |
| Anomaly Detection | scikit-learn (Isolation Forest) |
| Visualization | Plotly |
| Application | Streamlit |
| Version Control | Git / GitHub |

---

## Project Structure

```
stock-dashboard/
├── app.py              # Streamlit app — UI, sidebar, layout
├── src/
│   ├── data.py         # OHLCV data fetching and cleaning via yfinance
│   ├── features.py     # Technical indicator computation (SMA, EMA, RSI, Bollinger Bands)
│   ├── anomaly.py      # Isolation Forest anomaly detection pipeline
│   └── charts.py       # Plotly chart builders
└── requirements.txt
```

---

## How It Works

**1. Data Ingestion (`data.py`)**
Fetches historical OHLCV data for a given ticker and time period from Yahoo Finance. Handles multi-level column flattening, missing value removal, and datetime index formatting.

**2. Feature Engineering (`features.py`)**
Computes technical indicators on the closing price time series:
- **SMA** — simple moving average over 20 and 50 day windows
- **EMA** — exponential moving average weighted toward recent prices
- **RSI** — momentum oscillator measuring overbought/oversold conditions
- **Bollinger Bands** — volatility envelope 2 standard deviations above and below the SMA

**3. Anomaly Detection (`anomaly.py`)**
Runs an Isolation Forest model on a feature matrix of Close price, Volume, RSI, and Bollinger Band width. Isolation Forest isolates anomalies by randomly partitioning the feature space — unusual trading days require fewer partitions to isolate and receive lower anomaly scores. The contamination parameter controls the expected anomaly rate and is configurable in the UI.

**4. Visualization (`charts.py`)**
Builds interactive Plotly figures with a two-panel layout — candlestick chart with indicator overlays on top, RSI panel below. Anomalies are marked as red dots above the corresponding candle. Volume is rendered as a separate bar chart colored by daily price direction.

---

## Running Locally

```bash
git clone https://github.com/darylcarter2/stock-dashboard.git
cd stock-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Example

> AAPL — 1 year of price action with SMA, Bollinger Bands, RSI, and Isolation Forest anomaly markers

Anomalies flagged include high-volume earnings days, sharp RSI divergences, and unusual price movements outside the Bollinger Band range.

---

*Data sourced from Yahoo Finance via yfinance. For educational purposes only.*