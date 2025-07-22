# app.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("📈 NVDA Candlestick Chart")

# Get data
ticker = st.text_input("Enter Ticker", value="NVDA")
df = yf.download(ticker, period="1mo", interval="1d")

# Plot
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
fig.update_layout(title=f"{ticker} Candlestick", xaxis_title="Date", yaxis_title="Price")

st.plotly_chart(fig)
