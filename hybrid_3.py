
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_PATH = "/content/drive/MyDrive/market_ml_project-20260815T095833Z-1-001/market_ml_project"

INDEX_NAME = "SENSEX"       # NIFTY / BANKNIFTY / SENSEX
SPLIT_DATE = "2024-01-01"

GDELT_FILE = f"{PROJECT_PATH}/gdelt_weekly_sentiment.pkl"

PROB_THRESHOLD = 0.60

FILES = {
    "NIFTY": "nifty_weekly_labeled.csv",
    "BANKNIFTY": "banknifty_weekly_labeled.csv",
    "SENSEX": "sensex_weekly_labeled.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{FILES[INDEX_NAME]}"


# =============================================================================
# LOAD MARKET DATA
# =============================================================================

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])

df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print("=" * 70)
print(f"LOADED MARKET DATA — {INDEX_NAME}")
print("=" * 70)

print(f"Rows       : {df.shape[0]}")
print(f"Features   : {df.shape[1] - 1}")
print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
print(f"Labels     : {df['label'].value_counts().to_dict()}")


# =============================================================================
# TECHNICAL FEATURE ENGINEERING
# =============================================================================

print("\nCreating technical features...")


# -------------------------------------------------------------------------
# Price-derived features
# -------------------------------------------------------------------------

df["price_range"] = (
    (df["High"] - df["Low"]) /
    (df["Close"] + 1e-9)
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


# -------------------------------------------------------------------------
# EMA slopes
# -------------------------------------------------------------------------

df["ema_20_slope"] = df["ema_20"].pct_change(2)

df["ema_50_slope"] = df["ema_50"].pct_change(4)


# -------------------------------------------------------------------------
# EMA crossover
# -------------------------------------------------------------------------

df["ema_20_50_cross"] = (
    df["ema_20"] > df["ema_50"]
).astype(int)

df["ema_50_100_cross"] = (
    df["ema_50"] > df["ema_100"]
).astype(int)


# -------------------------------------------------------------------------
# RSI zones
# -------------------------------------------------------------------------

df["rsi_overbought"] = (
    df["rsi_14"] > 70
).astype(int)

df["rsi_oversold"] = (
    df["rsi_14"] < 30
).astype(int)

df["rsi_mid"] = (
    (df["rsi_14"] >= 45) &
    (df["rsi_14"] <= 55)
).astype(int)


# -------------------------------------------------------------------------
# MACD
# -------------------------------------------------------------------------

df["macd_cross"] = (
    df["macd"] > df["macd_signal"]
).astype(int)

df["macd_cross_change"] = df["macd_cross"].diff()


# -------------------------------------------------------------------------
# Bollinger Band squeeze
# -------------------------------------------------------------------------

df["bb_squeeze"] = (
    df["bb_width"] <
    df["bb_width"].rolling(20).quantile(0.2)
).astype(int)


# -------------------------------------------------------------------------
# ADX / DI
# -------------------------------------------------------------------------

df["trend_strong"] = (
    df["adx"] > 25
).astype(int)

df["di_cross"] = (
    df["plus_di"] > df["minus_di"]
).astype(int)


# -------------------------------------------------------------------------
# Momentum
# -------------------------------------------------------------------------

df["return_1w"] = df["Close"].pct_change(1)

df["return_2w"] = df["Close"].pct_change(2)

df["return_4w"] = df["Close"].pct_change(4)

df["return_8w"] = df["Close"].pct_change(8)

df["return_12w"] = df["Close"].pct_change(12)


# -------------------------------------------------------------------------
# Momentum acceleration
# -------------------------------------------------------------------------

df["mom_accel"] = (
    df["return_1w"] -
    df["return_2w"]
)


# -------------------------------------------------------------------------
# Previous labels
# IMPORTANT:
# These use ONLY previous weeks.
# -------------------------------------------------------------------------

df["label_lag1"] = df["label"].shift(1)

df["label_lag2"] = df["label"].shift(2)

df["label_lag3"] = df["label"].shift(3)


# -------------------------------------------------------------------------
# LEAKAGE-FREE consecutive UP weeks
#
# Original code used current label.
# This version uses label.shift(1), so the current target
# is NOT used to construct the feature.
# -------------------------------------------------------------------------

previous_label = df["label"].shift(1)

group_id = (
    previous_label != previous_label.shift()
).cumsum()

df["consec_up"] = (
    previous_label
    .groupby(group_id)
    .cumcount()
)

df["consec_up"] = (
    df["consec_up"] *
    previous_label
)


# -------------------------------------------------------------------------
# OBV momentum
# -------------------------------------------------------------------------

df["obv_momentum"] = (
    df["obv"].pct_change(4)
)


# -------------------------------------------------------------------------
# Volatility regime
# -------------------------------------------------------------------------

df["vol_20w"] = (
    df["return_1w"].rolling(20).std()
)

df["vol_ratio"] = (
    df["return_1w"].rolling(4).std() /
    (df["vol_20w"] + 1e-9)
)


# -------------------------------------------------------------------------
# RSI divergence proxy
# -------------------------------------------------------------------------

df["price_up_rsi_down"] = (
    (df["Close"] > df["Close"].shift(1)) &
    (df["rsi_14"] < df["rsi_14"].shift(1))
).astype(int)


# =============================================================================
# LOAD GDELT WEEKLY SENTIMENT
# =============================================================================

print("\n" + "=" * 70)
print("LOADING GDELT SENTIMENT")
print("=" * 70)

gdelt = pd.read_pickle(GDELT_FILE)

print(f"GDELT rows: {len(gdelt)}")

print("\nGDELT columns:")
print(gdelt.columns.tolist())


# =============================================================================
# GDELT DATE ALIGNMENT
# =============================================================================

# GDELT week format:
# 2015-02-14/2015-02-20
#
# We use the WEEK END date because the market prediction is for
# the following week.

if gdelt["week"].dtype == "object":

    gdelt["week_end"] = pd.to_datetime(
        gdelt["week"]
        .astype(str)
        .str.split("/")
        .str[-1]
    )

else:

    gdelt["week_end"] = pd.to_datetime(
        gdelt["week"]
    )


gdelt.set_index("week_end", inplace=True)

gdelt.sort_index(inplace=True)


# =============================================================================
# SENTIMENT FEATURES
# =============================================================================

print("\nCreating GDELT sentiment features...")


# Original GDELT features

gdelt["sentiment_change"] = (
    gdelt["mean_sentiment"].diff()
)

gdelt["sentiment_2w_change"] = (
    gdelt["mean_sentiment"].diff(2)
)

gdelt["sentiment_4w_avg"] = (
    gdelt["mean_sentiment"]
    .rolling(4)
    .mean()
)

gdelt["sentiment_4w_std"] = (
    gdelt["mean_sentiment"]
    .rolling(4)
    .std()
)

gdelt["news_count_change"] = (
    gdelt["news_count"].pct_change()
)

gdelt["news_count_4w_avg"] = (
    gdelt["news_count"]
    .rolling(4)
    .mean()
)


# Optional sentiment direction

gdelt["sentiment_positive"] = (
    gdelt["mean_sentiment"] > 0
).astype(int)

gdelt["sentiment_negative"] = (
    gdelt["mean_sentiment"] < 0
).astype(int)


# =============================================================================
# SELECT GDELT FEATURES
# =============================================================================

sentiment_cols = [
    "mean_sentiment",
    "median_sentiment",
    "sentiment_std",
    "news_count",

    "sentiment_change",
    "sentiment_2w_change",
    "sentiment_4w_avg",
    "sentiment_4w_std",

    "news_count_change",
    "news_count_4w_avg",

    "sentiment_positive",
    "sentiment_negative"
]

gdelt_features = gdelt[sentiment_cols].copy()


# =============================================================================
# MERGE MARKET + GDELT
# =============================================================================

print("\n" + "=" * 70)
print("MERGING MARKET DATA + GDELT")
print("=" * 70)

print(
    f"Market date range : "
    f"{df.index.min().date()} → {df.index.max().date()}"
)

print(
    f"GDELT date range  : "
    f"{gdelt_features.index.min().date()} → "
    f"{gdelt_features.index.max().date()}"
)


# Inner join ensures that only weeks having both market and sentiment
# information are used.

df = df.join(
    gdelt_features,
    how="inner"
)

print(f"\nRows after market/GDELT merge: {len(df)}")


# =============================================================================
# DROP NaN
# =============================================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

print(f"Rows after feature cleaning: {len(df)}")

print(
    f"Final date range: "
    f"{df.index.min().date()} → {df.index.max().date()}"
)


# =============================================================================
# FEATURE SELECTION
# =============================================================================

DROP_COLS = [
    "label",
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


X = df[feature_cols].copy()

y = df["label"].copy()


# Numerical safety

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.astype("float64")

X = X.clip(
    lower=-1e6,
    upper=1e6
)


# =============================================================================
# IDENTIFY FEATURE GROUPS
# =============================================================================

technical_features = [
    c for c in feature_cols
    if c not in sentiment_cols
]

sentiment_features = [
    c for c in feature_cols
    if c in sentiment_cols
]


print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(f"Total features     : {len(feature_cols)}")
print(f"Technical features : {len(technical_features)}")
print(f"Sentiment features : {len(sentiment_features)}")

print("\nSentiment features:")
for f in sentiment_features:
    print("  ", f)


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

X_train = X[X.index < SPLIT_DATE]

X_test = X[X.index >= SPLIT_DATE]

y_train = y[y.index < SPLIT_DATE]

y_test = y[y.index >= SPLIT_DATE]


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Train: {len(X_train)} weeks | "
    f"{X_train.index.min().date()} → "
    f"{X_train.index.max().date()}"
)

print(
    f"Test : {len(X_test)} weeks | "
    f"{X_test.index.min().date()} → "
    f"{X_test.index.max().date()}"
)

print(
    f"Train labels: "
    f"{y_train.value_counts().to_dict()}"
)

print(
    f"Test labels : "
    f"{y_test.value_counts().to_dict()}"
)


# =============================================================================
# RANDOM FOREST
# =============================================================================

rf = RandomForestClassifier(

    n_estimators=1000,

    max_depth=8,

    min_samples_leaf=5,

    min_samples_split=10,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1,

    oob_score=True
)

rf.fit(
    X_train,
    y_train
)


print("\nRF OOB Score:", round(rf.oob_score_, 4))


# =============================================================================
# GRADIENT BOOSTING
# =============================================================================

gbc = GradientBoostingClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.8,

    min_samples_leaf=10,

    random_state=42
)

gbc.fit(
    X_train,
    y_train
)


# =============================================================================
# SOFT VOTING ENSEMBLE
# =============================================================================

ensemble = VotingClassifier(

    estimators=[
        ("rf", rf),
        ("gbc", gbc)
    ],

    voting="soft",

    weights=[0.5, 0.5]
)

ensemble.fit(
    X_train,
    y_train
)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("HYBRID MODEL COMPARISON")
print("=" * 70)


models = [
    ("Random Forest", rf),
    ("Gradient Boosting", gbc),
    ("Hybrid Ensemble", ensemble)
]


for name, model in models:

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print(f"\n{name}")

    print(
        f"Accuracy: {acc:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )


# =============================================================================
# ENSEMBLE DETAILED REPORT
# =============================================================================

y_pred = ensemble.predict(
    X_test
)

y_proba = ensemble.predict_proba(
    X_test
)[:, 1]


print("\n" + "=" * 70)
print("DETAILED HYBRID REPORT")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print(
    "\nClassification Report:"
)

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

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =============================================================================
# HIGH-CONFIDENCE ANALYSIS
# =============================================================================

results = X_test.copy()

results["actual"] = y_test.values

results["prob_up"] = y_proba

results["predicted"] = y_pred


print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ANALYSIS")
print("=" * 70)


for threshold in [
    0.55,
    0.60,
    0.65,
    0.70
]:

    hc = results[
        results["prob_up"] >= threshold
    ]

    if len(hc) > 0:

        acc = accuracy_score(
            hc["actual"],
            hc["predicted"]
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"{len(hc)} weeks | "
            f"accuracy {acc:.4f}"
        )

    else:

        print(
            f"Threshold {threshold:.2f}: "
            f"No predictions"
        )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance.head(30).round(4)
)


# =============================================================================
# TECHNICAL VS SENTIMENT IMPORTANCE
# =============================================================================

technical_importance = (
    importance[
        importance.index.isin(
            technical_features
        )
    ]
    .sum()
)

sentiment_importance = (
    importance[
        importance.index.isin(
            sentiment_features
        )
    ]
    .sum()
)


print("\n" + "=" * 70)
print("FEATURE GROUP IMPORTANCE")
print("=" * 70)

print(
    f"Technical feature importance : "
    f"{technical_importance:.4f}"
)

print(
    f"Sentiment feature importance : "
    f"{sentiment_importance:.4f}"
)

print(
    f"Technical percentage : "
    f"{technical_importance * 100:.2f}%"
)

print(
    f"Sentiment percentage : "
    f"{sentiment_importance * 100:.2f}%"
)


print("\nTop sentiment features:")

sentiment_importance_series = (
    importance[
        importance.index.isin(
            sentiment_features
        )
    ]
    .sort_values(
        ascending=False
    )
)

print(
    sentiment_importance_series.round(4)
)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION — HYBRID RF")
print("=" * 70)


tscv = TimeSeriesSplit(
    n_splits=5
)

wf_acc = []


for fold, (train_idx, test_idx) in enumerate(
    tscv.split(X)
):

    Xtr = X.iloc[train_idx]

    Xte = X.iloc[test_idx]

    ytr = y.iloc[train_idx]

    yte = y.iloc[test_idx]


    fold_rf = RandomForestClassifier(

        n_estimators=500,

        max_depth=8,

        min_samples_leaf=5,

        min_samples_split=10,

        max_features="sqrt",

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )


    fold_rf.fit(
        Xtr,
        ytr
    )


    fold_pred = fold_rf.predict(
        Xte
    )


    fold_acc = accuracy_score(
        yte,
        fold_pred
    )


    wf_acc.append(
        fold_acc
    )


    print(
        f"Fold {fold + 1}: "
        f"{len(Xtr)} train | "
        f"{len(Xte)} test | "
        f"accuracy {fold_acc:.4f}"
    )


print(
    f"\nWalk-forward mean: "
    f"{np.mean(wf_acc):.4f}"
)

print(
    f"Walk-forward std : "
    f"{np.std(wf_acc):.4f}"
)


# =============================================================================
# BASELINE
# =============================================================================

majority_class = y_test.mode()[0]

baseline_accuracy = (
    y_test == majority_class
).mean()


print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(
    f"Majority baseline: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Hybrid accuracy  : "
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Difference        : "
    f"{accuracy_score(y_test, y_pred) - baseline_accuracy:+.4f}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(f"FINAL HYBRID SUMMARY — {INDEX_NAME}")
print("=" * 70)

print(
    f"Features used             : {X.shape[1]}"
)

print(
    f"Technical features        : {len(technical_features)}"
)

print(
    f"Sentiment features        : {len(sentiment_features)}"
)

print(
    f"Train size                : {len(X_train)} weeks"
)

print(
    f"Test size                 : {len(X_test)} weeks"
)

print(
    f"RF Accuracy               : "
    f"{accuracy_score(y_test, rf.predict(X_test)):.4f}"
)

print(
    f"GBC Accuracy              : "
    f"{accuracy_score(y_test, gbc.predict(X_test)):.4f}"
)

print(
    f"Hybrid Ensemble Accuracy  : "
    f"{accuracy_score(y_test, ensemble.predict(X_test)):.4f}"
)

print(
    f"Walk-forward Mean         : "
    f"{np.mean(wf_acc):.4f}"
)

print(
    f"Walk-forward Std          : "
    f"{np.std(wf_acc):.4f}"
)

print(
    f"RF OOB Score              : "
    f"{rf.oob_score_:.4f}"
)

print(
    f"Technical importance      : "
    f"{technical_importance * 100:.2f}%"
)

print(
    f"Sentiment importance      : "
    f"{sentiment_importance * 100:.2f}%"
)

print("=" * 70)

