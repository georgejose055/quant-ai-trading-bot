import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ─── Fetch Data ───────────────────────────────────────────────────────────────
def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # ── Direct key access (works on ALL yfinance versions) ────────────────
    try:
        df = pd.DataFrame({
            "Open":   raw["Open"].squeeze(),
            "High":   raw["High"].squeeze(),
            "Low":    raw["Low"].squeeze(),
            "Close":  raw["Close"].squeeze(),
            "Volume": raw["Volume"].squeeze(),
        }, index=raw.index)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)
        if len(df) < 60:
            raise ValueError(f"Not enough data for {ticker} (only {len(df)} rows)")
        return df
    except KeyError:
        pass

    # ── Fallback: MultiIndex level detection ──────────────────────────────
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(v).strip() for v in df.columns.get_level_values(0).unique()]
        l1 = [str(v).strip() for v in df.columns.get_level_values(1).unique()]
        ohlcv = {"Open", "High", "Low", "Close", "Volume"}
        if any(v in ohlcv for v in l0):
            df.columns = df.columns.get_level_values(0)
        elif any(v in ohlcv for v in l1):
            df.columns = df.columns.get_level_values(1)
        else:
            # Last resort: alphabetical rename (yfinance returns C,H,L,O,V order)
            df.columns = df.columns.droplevel(1)
            if len(df.columns) == 5:
                df.columns = ["Close", "High", "Low", "Open", "Volume"]

    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.rename(columns={"Adj Close": "Close", "Adj_Close": "Close"})
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' for {ticker}. "
                f"Available: {df.columns.tolist()}"
            )
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df


# ─── Indicator Helpers ────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s      = series.squeeze()
    delta  = s.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    s      = series.squeeze()
    ema12  = s.ewm(span=12, adjust=False).mean()
    ema26  = s.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_bollinger(series: pd.Series, window: int = 20):
    s   = series.squeeze()
    ma  = s.rolling(window).mean()
    std = s.rolling(window).std()
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
    k      = 100 * (df["Close"] - low_n) / (high_n - low_n + 1e-9)
    d      = k.rolling(3).mean()
    return k, d


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = df["Close"].diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    return (df["Volume"] * direction).cumsum()


# ─── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure clean flat columns
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            vals = [str(v).strip() for v in df.columns.get_level_values(level)]
            if "Close" in vals:
                df.columns = df.columns.get_level_values(level)
                break
        else:
            df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.rename(columns={"Adj Close": "Close", "Adj_Close": "Close"})
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    close = df["Close"].squeeze()

    # ── Returns ───────────────────────────────────────────────────────────
    df["Return"]    = close.pct_change()
    df["Return_2"]  = close.pct_change(2)
    df["Return_5"]  = close.pct_change(5)
    df["Return_10"] = close.pct_change(10)
    df["ROC_5"]     = close.pct_change(5)  * 100
    df["ROC_10"]    = close.pct_change(10) * 100

    # ── Moving Averages ───────────────────────────────────────────────────
    df["MA5"]  = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # ── MA Crossovers ─────────────────────────────────────────────────────
    df["MA5_20_cross"]  = (df["MA5"]  > df["MA20"]).astype(int)
    df["MA10_50_cross"] = (df["MA10"] > df["MA50"]).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────
    df["Volatility"]  = df["Return"].rolling(10).std()
    df["Volatility5"] = df["Return"].rolling(5).std()

    # ── Volume ────────────────────────────────────────────────────────────
    df["Volume_MA"]    = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-9)
    df["OBV"]          = compute_obv(df)
    df["OBV_MA"]       = df["OBV"].rolling(10).mean()

    # ── RSI ───────────────────────────────────────────────────────────────
    df["RSI"]      = compute_rsi(close, 14)
    df["RSI_fast"] = compute_rsi(close, 7)

    # ── MACD ──────────────────────────────────────────────────────────────
    df["MACD"], df["MACD_Signal"] = compute_macd(close)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────
    df["BB_Upper"], df["BB_Lower"] = compute_bollinger(close)
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / (df["MA20"] + 1e-9)
    df["BB_Position"] = (close - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"] + 1e-9
    )

    # ── ATR ───────────────────────────────────────────────────────────────
    df["ATR"] = compute_atr(df)

    # ── Stochastic ────────────────────────────────────────────────────────
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic(df)

    # ── Williams %R ───────────────────────────────────────────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Williams_R"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    # ── High/Low Channel ──────────────────────────────────────────────────
    df["High_5"]      = df["High"].rolling(5).max()
    df["Low_5"]       = df["Low"].rolling(5).min()
    df["Channel_Pos"] = (close - df["Low_5"]) / (
        df["High_5"] - df["Low_5"] + 1e-9
    )

    # ── Target: ±2% move over next 3 days ────────────────────────────────
    future_return = (close.shift(-3) - close) / close
    df["Target"]  = np.where(future_return > 0.02, 1, 0)
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

    X = df[features].copy()

    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(0)
    for col in X.columns:
        if isinstance(X[col], pd.DataFrame):
            X[col] = X[col].iloc[:, 0]

    X = X.astype(float)
    y = df["Target"].astype(int)

    split   = int(len(df) * 0.80)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

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
    X  = df[features].copy()

    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(0)
    for col in X.columns:
        if isinstance(X[col], pd.DataFrame):
            X[col] = X[col].iloc[:, 0]

    X = scaler.transform(X.astype(float))
    df["Prediction"] = model.predict(X)
    df["Signal"]     = df["Prediction"].map({1: "BUY 🟢", 0: "SELL 🔴"})
    return df