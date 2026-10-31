import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# CONFIG
# ============================================================

PROJECT_PATH = "/content/drive/MyDrive/market_ml_project-20260815T095833Z-1-001/market_ml_project"

INDEX_NAME = "NIFTY"       # NIFTY / BANKNIFTY / SENSEX
SPLIT_DATE = "2024-01-01"

FILES = {
    "NIFTY": "nifty_daily_features.csv",
    "BANKNIFTY": "banknifty_daily_features.csv",
    "SENSEX": "sensex_daily_features.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{FILES[INDEX_NAME]}"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["Date"]
)

df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"Loaded {INDEX_NAME}")
print(f"Rows: {len(df)}")
print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")


# ============================================================
# CREATE 5-DAY TARGET
# ============================================================

df["future_close_5d"] = df["Close"].shift(-5)

df["future_return_5d"] = (
    df["future_close_5d"] - df["Close"]
) / df["Close"]

df["label"] = (
    df["future_close_5d"] > df["Close"]
).astype(int)


# ============================================================
# TECHNICAL FEATURE ENGINEERING
# ============================================================

df["price_range"] = (
    (df["High"] - df["Low"]) /
    df["Close"]
)

df["close_position"] = (
    (df["Close"] - df["Low"]) /
    (df["High"] - df["Low"] + 1e-9)
)

df["gap"] = (
    df["Open"] /
    df["Close"].shift(1)
    - 1
)

# EMA
df["ema20_slope"] = df["ema_20"].pct_change(2)
df["ema50_slope"] = df["ema_50"].pct_change(5)

df["ema20_50_cross"] = (
    df["ema_20"] > df["ema_50"]
).astype(int)

df["ema50_100_cross"] = (
    df["ema_50"] > df["ema_100"]
).astype(int)

# RSI
df["rsi_overbought"] = (
    df["rsi_14"] > 70
).astype(int)

df["rsi_oversold"] = (
    df["rsi_14"] < 30
).astype(int)

# MACD
df["macd_cross"] = (
    df["macd"] > df["macd_signal"]
).astype(int)

df["macd_cross_change"] = (
    df["macd_cross"].diff()
)

# ADX
df["trend_strong"] = (
    df["adx"] > 25
).astype(int)

df["di_cross"] = (
    df["plus_di"] > df["minus_di"]
).astype(int)

# Momentum
df["return_1d"] = df["Close"].pct_change(1)
df["return_3d"] = df["Close"].pct_change(3)
df["return_5d"] = df["Close"].pct_change(5)
df["return_10d"] = df["Close"].pct_change(10)

# Volatility
df["volatility_10d"] = (
    df["return_1d"].rolling(10).std()
)

df["volatility_20d"] = (
    df["return_1d"].rolling(20).std()
)

df["vol_ratio"] = (
    df["volatility_10d"] /
    (df["volatility_20d"] + 1e-9)
)

# OBV
df["obv_momentum"] = (
    df["obv"].pct_change(5)
)

df.dropna(inplace=True)


# ============================================================
# FEATURES / LABEL
# ============================================================

DROP_COLS = [
    "label",
    "future_close_5d",
    "future_return_5d",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS
]

X = df[feature_cols]
y = df["label"]

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

print(f"\nTotal features: {X.shape[1]}")
print(f"Label distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train = X[X.index < SPLIT_DATE]
X_test = X[X.index >= SPLIT_DATE]

y_train = y[y.index < SPLIT_DATE]
y_test = y[y.index >= SPLIT_DATE]

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION — 5 DAY TECHNICAL")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Bearish",
            "Bullish"
        ]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("WALK-FORWARD VALIDATION")
print("=" * 60)

tscv = TimeSeriesSplit(
    n_splits=5
)

scores = []

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(X),
    start=1
):

    fold_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight="balanced",
            random_state=42
        ))
    ])

    fold_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    pred = fold_model.predict(
        X.iloc[test_idx]
    )

    score = accuracy_score(
        y.iloc[test_idx],
        pred
    )

    scores.append(score)

    print(
        f"Fold {fold}: "
        f"{score:.4f}"
    )

print(
    f"\nMean accuracy: "
    f"{np.mean(scores):.4f}"
)

print(
    f"Std: "
    f"{np.std(scores):.4f}"
)