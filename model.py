import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ─── Fetch Data ───────────────────────────────────────────────────────────────
def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # Fix yfinance MultiIndex columns (v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = [str(c).strip().title() for c in df.columns]

    # Rename variants
    df = df.rename(columns={"Adj Close": "Close", "Adj_Close": "Close"})

    # Validate columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Available: {df.columns.tolist()}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df


# ─── Indicator Helpers ─────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_bollinger(series: pd.Series, window: int = 20):
    ma  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + 2 * std, ma - 2 * std


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_stochastic(df: pd.DataFrame, period: int = 14):
    low_n  = df["Low"].rolling(period).min()
    high_n = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_n) / (high_n - low_n + 1e-9)
    d = k.rolling(3).mean()
    return k, d


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (df["Volume"] * direction).cumsum()


# ─── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Returns ───────────────────────────────────────────────────────────────
    df["Return"]     = df["Close"].pct_change()
    df["Return_2"]   = df["Close"].pct_change(2)
    df["Return_5"]   = df["Close"].pct_change(5)
    df["Return_10"]  = df["Close"].pct_change(10)
    df["ROC_5"]      = df["Close"].pct_change(5)  * 100
    df["ROC_10"]     = df["Close"].pct_change(10) * 100

    # ── Moving Averages ────────────────────────────────────────────────────────
    df["MA5"]        = df["Close"].rolling(5).mean()
    df["MA10"]       = df["Close"].rolling(10).mean()
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()

    # ── MA Crossovers ──────────────────────────────────────────────────────────
    df["MA5_20_cross"]  = (df["MA5"]  > df["MA20"]).astype(int)
    df["MA10_50_cross"] = (df["MA10"] > df["MA50"]).astype(int)

    # ── Volatility ─────────────────────────────────────────────────────────────
    df["Volatility"]  = df["Return"].rolling(10).std()
    df["Volatility5"] = df["Return"].rolling(5).std()

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["Volume_MA"]    = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-9)
    df["OBV"]          = compute_obv(df)
    df["OBV_MA"]       = df["OBV"].rolling(10).mean()

    # ── RSI ────────────────────────────────────────────────────────────────────
    df["RSI"]      = compute_rsi(df["Close"], 14)
    df["RSI_fast"] = compute_rsi(df["Close"], 7)

    # ── MACD ───────────────────────────────────────────────────────────────────
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    df["BB_Upper"], df["BB_Lower"] = compute_bollinger(df["Close"])
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / (df["MA20"] + 1e-9)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ── ATR ────────────────────────────────────────────────────────────────────
    df["ATR"] = compute_atr(df)

    # ── Stochastic ─────────────────────────────────────────────────────────────
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic(df)

    # ── Williams %R ────────────────────────────────────────────────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Williams_R"] = -100 * (high14 - df["Close"]) / (high14 - low14 + 1e-9)

    # ── High/Low Channel ───────────────────────────────────────────────────────
    df["High_5"]     = df["High"].rolling(5).max()
    df["Low_5"]      = df["Low"].rolling(5).min()
    df["Channel_Pos"]= (df["Close"] - df["Low_5"]) / (df["High_5"] - df["Low_5"] + 1e-9)

    # ── Target: Only label STRONG moves (±2% over 3 days) ─────────────────────
    future_return    = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    df["Target"]     = np.where(future_return > 0.02, 1, 0)

    # Remove ambiguous rows (between -2% and +2%)
    df = df[future_return.abs() > 0.02]

    df.dropna(inplace=True)
    return df


# ─── Train Model ──────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    features = [
        "Return", "Return_2", "Return_5", "Return_10",
        "ROC_5", "ROC_10",
        "MA5", "MA10", "MA20", "MA50",
        "MA5_20_cross", "MA10_50_cross",
        "Volatility", "Volatility5",
        "Volume_Ratio", "OBV_MA",
        "RSI", "RSI_fast",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width", "BB_Position",
        "ATR", "Stoch_K", "Stoch_D",
        "Williams_R", "Channel_Pos"
    ]

    X = df[features].astype(float)
    y = df["Target"].astype(int)

    # Strict time-based split — no shuffle to avoid future leakage
    split   = int(len(df) * 0.80)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Tuned Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, features, scaler, round(acc * 100, 2)


# ─── Generate Signals ─────────────────────────────────────────────────────────
def get_signals(df: pd.DataFrame, model, features: list, scaler) -> pd.DataFrame:
    df = df.copy()
    X  = scaler.transform(df[features].astype(float))
    df["Prediction"] = model.predict(X)
    df["Signal"]     = df["Prediction"].map({1: "BUY 🟢", 0: "SELL 🔴"})
    return df