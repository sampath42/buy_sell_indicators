# app.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

import pandas as pd
import mplfinance as mpf
import matplotlib
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.title("Buy Sell Indicators")

# --------------- Pattern Detection Functions ---------------

def classify_single(o, h, l, c):
    body = abs(c - o)
    tr = h - l
    if tr == 0: return "Doji (Flat)"
    uw = h - max(o, c)
    lw = min(o, c) - l
    br = body / tr
    if br <= 0.1:
        return "Doji"
    elif br <= 0.3 and abs(uw - lw) / tr <= 0.2:
        return "Spinning Top"
    elif c > o:
        if lw >= 2 * body and uw <= 0.1 * tr:
            return "Hammer"
        elif uw >= 2 * body and lw <= 0.1 * tr:
            return "Inverted Hammer"
    elif o > c:
        if uw >= 2 * body and lw <= 0.1 * tr:
            return "Shooting Star"
    if o == l and c == h:
        return "Bullish Marubozu"
    elif o == h and c == l:
        return "Bearish Marubozu"
    return None

def classify_two(df, i):
    o1, c1 = df.iloc[i-1][['Open', 'Close']]
    o2, c2 = df.iloc[i][['Open', 'Close']]
    if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
        return "Bullish Engulfing"
    if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
        return "Bearish Engulfing"
    return None

def classify_three(df, i):
    o1, c1 = df.iloc[i-2][['Open', 'Close']]
    o2, c2 = df.iloc[i-1][['Open', 'Close']]
    o3, c3 = df.iloc[i][['Open', 'Close']]
    if c1 < o1 and abs(c2 - o2) < abs(c1 - o1)*0.3 and c3 > o3 and c3 > (o1 + c1) / 2:
        return "Morning Star"
    if c1 > o1 and abs(c2 - o2) < abs(c1 - o1)*0.3 and c3 < o3 and c3 < (o1 + c1) / 2:
        return "Evening Star"
    if all(df.iloc[j]['Close'] > df.iloc[j]['Open'] for j in [i-2, i-1, i]) and \
       all(df.iloc[j]['Close'] > df.iloc[j-1]['Close'] for j in [i-1, i]):
        return "Three White Soldiers"
    if all(df.iloc[j]['Close'] < df.iloc[j]['Open'] for j in [i-2, i-1, i]) and \
       all(df.iloc[j]['Close'] < df.iloc[j-1]['Close'] for j in [i-1, i]):
        return "Three Black Crows"
    return None

# --------------- Fetch Data ---------------
#NVDA250725P00170000
ticker = st.text_input("Enter Ticker", value="NVDA")
period = st.text_input("Enter Period", value="30d")
interval = st.text_input("Enter Interval", value="5m")
macd = st.checkbox("MACD")
stochRSI = st.checkbox("StochRSI")
stochOsc = st.checkbox("StochOsc")

df = yf.download(ticker, period=period, interval=interval)
df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)

# --------------- Pattern Detection ---------------
df['Pattern'] = None
for i in range(len(df)):
    pattern = classify_single(df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i])
    if i >= 1:
        two = classify_two(df, i)
        if two: pattern = (pattern + ", " if pattern else "") + two
    if i >= 2:
        three = classify_three(df, i)
        if three: pattern = (pattern + ", " if pattern else "") + three
    df.iat[i, df.columns.get_loc('Pattern')] = pattern
    
# ------------------Bollinger-------------------
df['MA20'] = df['Close'].rolling(window=20).mean()
df['Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

bollinger_apds = [
    mpf.make_addplot(df['Upper'], color='gray'),
    mpf.make_addplot(df['MA20'], color='blue'),
    mpf.make_addplot(df['Lower'], color='gray')
]

# MACD = EMA(12) - EMA(26)
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Histogram'] = df['MACD'] - df['Signal']

macd_apds = [
    mpf.make_addplot(df['MACD'], panel=1, color='blue', ylabel='MACD'),
    mpf.make_addplot(df['Signal'], panel=1, color='orange'),
    mpf.make_addplot(df['Histogram'], panel=1, type='bar', color='gray', alpha=0.5)
]

#---------------Stoch Oscilator-------------
# Set the lookback period (typically 14)
low_min = df['Low'].rolling(window=14).min()
high_max = df['High'].rolling(window=14).max()

# %K line
df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)

# %D line (3-period moving average of %K)
df['%D'] = df['%K'].rolling(window=3).mean()

upper_bound = pd.Series(80, index=df.index)
lower_bound = pd.Series(20, index=df.index)

stoch_apds = [
    mpf.make_addplot(df['%K'], panel=2, color='purple', ylabel='Stoch'),
    mpf.make_addplot(df['%D'], panel=2, color='green'),
    mpf.make_addplot(upper_bound, panel=2, color='red', linestyle='-'),  # Overbought line
    mpf.make_addplot(lower_bound, panel=2, color='blue', linestyle='-')  # Oversold line
]

# Compute classic RSI first
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Now compute Stochastic RSI
rsi_min = df['RSI'].rolling(window=14).min()
rsi_max = df['RSI'].rolling(window=14).max()

df['StochRSI_K'] = 100 * (df['RSI'] - rsi_min) / (rsi_max - rsi_min)
df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean()

#--------------------Buy Sell Strategy--------------
# Shift %K and %D to compare with previous value
prev_k = df['%K'].shift(1)
prev_d = df['%D'].shift(1)

# Detect crossover (%K rises above %D)
crossover = (df['%K'] > df['%D'])

# Crossover condition: %K falls below %D
stoch_cross_down = (df['%K'] < df['%D'])

# Confirm %D is above 20 (escaping oversold)
oversold_escape = df['%D'] > 20

# Confirm overbought state
overbought_zone = df['%D'] > 80

# Confirm Doji pattern
is_doji = df['Pattern'].str.contains("Doji", na=False)

# Combine all conditions
buy_condition = crossover & oversold_escape #& is_doji
buy_signals = df[buy_condition]

# Final sell condition
sell_condition = stoch_cross_down & overbought_zone #& has_bearish_pattern
sell_signals = df[sell_condition]

#buy_signals = df[df['Pattern'].str.contains("Bullish Engulfing", na=False)]
buy_x = buy_signals.index
buy_y = buy_signals['Low'] - 0.05  # slight offset below candle

sell_x = sell_signals.index
sell_y = sell_signals['High'] + 0.05  # offset above candle

#--------------Plot----------------------
fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                    #row_heights=[0.45, 0.2, 0.2, 0.15],
                    #vertical_spacing=0.03,
                    #subplot_titles=("Candlestick", "MACD", "Stochastic Oscillator", "Stoch RSI")
                   )

# Row 1: Candlesticks + Bollinger Bands
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name='Candles',
                             hovertext=
                                    df['Open']
                            ), 
                              row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue'), name='MA20'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray'), name='Upper'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray'), name='Lower'), row=1, col=1)

# Buy markers (green upward triangles)
fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers',
                         marker=dict(symbol='triangle-up', color='green', size=8),
                         hovertext=buy_signals['Pattern'],
                         hoverinfo='text+x+y',
                         name='Buy Signal'), row=1, col=1)


fig.add_trace(go.Scatter(
    x=sell_x,
    y=sell_y,
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=8),
    hovertext=buy_signals['Pattern'],
    hoverinfo='text+x+y',
    name='Sell Signal'
), row=1, col=1)

# Row 2: MACD
#fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=2, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange'), name='Signal'), row=2, col=1)
#fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], marker_color='gray', name='Histogram'), row=2, col=1)

# Row 3: Stochastic Oscillator
#fig.add_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='purple'), name='%K'), row=3, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='green'), name='%D'), row=3, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=[80]*len(df), line=dict(color='red', dash='dash'), name='Overbought'), row=3, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=[20]*len(df), line=dict(color='blue', dash='dash'), name='Oversold'), row=3, col=1)

# Row 4: StochRSI
#fig.add_trace(go.Scatter(x=df.index, y=df['StochRSI_K'], line=dict(color='purple'), name='StochRSI_K'), row=4, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=df['StochRSI_D'], line=dict(color='green'), name='StochRSI_D'), row=4, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=[80]*len(df), line=dict(color='red', dash='dash'), name='Overbought'), row=4, col=1)
#fig.add_trace(go.Scatter(x=df.index, y=[20]*len(df), line=dict(color='blue', dash='dash'), name='Oversold'), row=4, col=1)

fig.update_layout(title='NVDA',
                  xaxis_rangeslider_visible=False,
                  height=1000)

# Layout enhancements
fig.update_layout(title='NVDA',
                  height=900,
                  xaxis_rangeslider_visible=False)

fig.update_layout(
    autosize=True,
    width=None,  # Leave unset to allow container to control width
    height=1000,  # You can increase this to fill more vertical space
    margin=dict(l=20, r=20, t=40, b=20)
)


fig.show()


st.plotly_chart(fig)
st.set_page_config(layout="wide")
