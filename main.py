import streamlit as st
from newsapi import NewsApiClient
from textblob import TextBlob
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='685892605c6546309812cd1add440db3')

# Indian Stock Options by Sector
stock_options = {
    "IT": {"TCS": "TCS.NS", "Infosys": "INFY.NS", "Wipro": "WIPRO.NS"},
    "Banking": {"HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS"},
    "Pharma": {"Sun Pharma": "SUNPHARMA.NS", "Dr. Reddy's": "DRREDDY.NS", "Cipla": "CIPLA.NS"},
    "Energy": {"Reliance": "RELIANCE.NS", "ONGC": "ONGC.NS", "Power Grid": "POWERGRID.NS"}
}

# Fetch News
@st.cache_data(ttl=600)
def fetch_news(stock_name, num_articles):
    try:
        articles = newsapi.get_everything(q=stock_name, language='en', sort_by='publishedAt', page_size=num_articles)
        return articles['articles']
    except:
        return []

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    else:
        return 'neutral', polarity

# Load Stock Data
@st.cache_data(ttl=3600)
def load_stock_data(stock_symbol, period='6mo', interval='1d'):
    stock = yf.Ticker(stock_symbol)
    return stock.history(period=period, interval=interval)

# Feature Engineering
def prepare_data(data):
    data['Open-Close'] = data['Open'] - data['Close']
    data['High-Low'] = data['High'] - data['Low']
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data.dropna()

# Train Models
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    lgbm = lgb.LGBMClassifier(n_estimators=500, max_depth=10, random_state=42)
    lgbm.fit(X_train, y_train)

    cat = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, verbose=0, random_state=42)
    cat.fit(X_train, y_train)

    return rf, lgbm, cat

# Predict Function
def predict(models, X_test):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_test)
    return preds

# Plotting Functions
def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(title="Candlestick Chart", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_moving_averages(data):
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50'))
    fig.update_layout(title="Moving Averages", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_volume(data):
    fig = px.bar(x=data.index, y=data['Volume'], labels={'x': 'Date', 'y': 'Volume'}, title='Volume', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(data):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=RSI, name='RSI'))
    fig.update_layout(title='Relative Strength Index (RSI)', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Predicted vs Actual Graph
def plot_predicted_vs_actual(y_test, y_pred, model_name):
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.legend()
    st.pyplot(plt)

# Streamlit App Starts
if 'entered' not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.title("📈 Stock Market Prediction System")
    st.write("Predict stock prices, visualize trends, and analyze news sentiment.")
    if st.button("🚀 Start"):
        st.session_state.entered = True
else:
    st.sidebar.title("📚 Settings")
    sector = st.sidebar.selectbox("Select Sector", list(stock_options.keys()))
    stock = st.sidebar.selectbox("Select Stock", list(stock_options[sector].keys()))
    stock_symbol = stock_options[sector][stock]

    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "LightGBM", "CatBoost"])
    graph_choice = st.sidebar.selectbox("Select Graph", ["Candlestick", "Moving Averages", "Volume", "RSI"])

    st.sidebar.subheader("⚙️ Advanced Settings")
    split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.2)
    show_pred_actual = st.sidebar.checkbox("Show Predicted vs Actual Prices", True)

    st.title(f"📈 Analysis for {stock}")

    data = load_stock_data(stock_symbol)

    if graph_choice == "Candlestick":
        plot_candlestick(data)
    elif graph_choice == "Moving Averages":
        plot_moving_averages(data)
    elif graph_choice == "Volume":
        plot_volume(data)
    elif graph_choice == "RSI":
        plot_rsi(data)

    data = prepare_data(data)
    X = data[['Open-Close', 'High-Low', 'Return']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    rf, lgbm, cat = train_models(X_train, y_train)
    models = {"Random Forest": rf, "LightGBM": lgbm, "CatBoost": cat}

    preds = predict(models, X_test)
    acc = accuracy_score(y_test, preds[model_choice])
    st.metric(label=f"{model_choice} Accuracy", value=f"{acc:.2%}")

    if show_pred_actual:
        plot_predicted_vs_actual(y_test, preds[model_choice], model_choice)

    st.write("---")
    st.subheader("📰 News & Sentiment Analysis")
    num_articles = st.slider("Number of News Articles", 3, 15, 5)

    with st.spinner('Fetching latest news...'):
        news = fetch_news(stock, num_articles)
        if news:
            for article in news:
                title = article['title']
                url = article['url']
                sentiment, _ = analyze_sentiment(title)
                display = f"**[{title}]({url})**"
                if sentiment == 'positive':
                    st.success(display)
                elif sentiment == 'negative':
                    st.error(display)
                else:
                    st.info(display)
                time.sleep(0.3)
        else:
            st.warning("No news found.")

    st.write("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
