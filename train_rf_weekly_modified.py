
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

# Change this to:
# NIFTY / BANKNIFTY / SENSEX
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent

INDEX_NAME = "BANKNIFTY"
SPLIT_DATE = "2024-01-01"
PROB_THRESHOLD = 0.60

FILES = {
    "NIFTY": "nifty_weekly_labeled.csv",
    "BANKNIFTY": "banknifty_weekly_labeled.csv",
    "SENSEX": "sensex_weekly_labeled.csv"
}

DATA_FILE = PROJECT_PATH / FILES[INDEX_NAME]

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["Date"]
)

df.set_index("Date", inplace=True)
df.sort_index(inplace=True)


print(f"Loaded {INDEX_NAME}: {df.shape[0]} rows | "
      f"{df.shape[1] - 1} features + label")

print(
    f"Date range: "
    f"{df.index[0].date()} → {df.index[-1].date()}"
)

print(
    f"Label balance: "
    f"{df['label'].value_counts().to_dict()}"
)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
#
# IMPORTANT:
# The target is:
#
# label = 1 if NEXT week's Close > CURRENT week's Close
#
# Therefore, every feature used for predicting the current row must be
# available at the CURRENT week's close or earlier.
#
# We must NOT use the current row's label as a feature.
# =============================================================================


# -----------------------------------------------------------------------------
# Price-derived features
# -----------------------------------------------------------------------------

df["price_range"] = (
    (df["High"] - df["Low"]) / df["Close"]
)

df["close_position"] = (
    (df["Close"] - df["Low"]) /
    (df["High"] - df["Low"] + 1e-9)
)

df["gap"] = (
    df["Open"] / df["Close"].shift(1) - 1
)


# -----------------------------------------------------------------------------
# EMA slopes
# -----------------------------------------------------------------------------

df["ema_20_slope"] = (
    df["ema_20"].pct_change(2)
)

df["ema_50_slope"] = (
    df["ema_50"].pct_change(4)
)


# -----------------------------------------------------------------------------
# EMA crossover signals
# -----------------------------------------------------------------------------

df["ema_20_50_cross"] = (
    df["ema_20"] > df["ema_50"]
).astype(int)

df["ema_50_100_cross"] = (
    df["ema_50"] > df["ema_100"]
).astype(int)


# -----------------------------------------------------------------------------
# RSI zones
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# MACD signals
# -----------------------------------------------------------------------------

df["macd_cross"] = (
    df["macd"] > df["macd_signal"]
).astype(int)

df["macd_cross_change"] = (
    df["macd_cross"].diff()
)


# -----------------------------------------------------------------------------
# Bollinger Band squeeze
# -----------------------------------------------------------------------------

df["bb_squeeze"] = (
    df["bb_width"] <
    df["bb_width"].rolling(20).quantile(0.20)
).astype(int)


# -----------------------------------------------------------------------------
# ADX / Directional movement
# -----------------------------------------------------------------------------

df["trend_strong"] = (
    df["adx"] > 25
).astype(int)

df["di_cross"] = (
    df["plus_di"] > df["minus_di"]
).astype(int)


# -----------------------------------------------------------------------------
# Multi-period momentum
# -----------------------------------------------------------------------------

df["return_1w"] = (
    df["Close"].pct_change(1)
)

df["return_2w"] = (
    df["Close"].pct_change(2)
)

df["return_4w"] = (
    df["Close"].pct_change(4)
)

df["return_8w"] = (
    df["Close"].pct_change(8)
)

df["return_12w"] = (
    df["Close"].pct_change(12)
)


# -----------------------------------------------------------------------------
# Momentum acceleration
# -----------------------------------------------------------------------------

df["mom_accel"] = (
    df["return_1w"] -
    df["return_2w"]
)


# =============================================================================
# HISTORICAL LABEL FEATURES
# =============================================================================
#
# IMPORTANT:
#
# Current label:
#     label[t] = direction of week t+1
#
# Therefore:
#
# label_lag1[t] = direction that occurred after week t-1
# label_lag2[t] = direction that occurred after week t-2
# label_lag3[t] = direction that occurred after week t-3
#
# These are historical information and do not contain label[t].
# =============================================================================


df["label_lag1"] = (
    df["label"].shift(1)
)

df["label_lag2"] = (
    df["label"].shift(2)
)

df["label_lag3"] = (
    df["label"].shift(3)
)


# =============================================================================
# LEAKAGE-FREE CONSECUTIVE BULLISH WEEKS
# =============================================================================
#
# OLD VERSION:
#
# df["consec_up"] = df["label"].groupby(...).cumcount()
#
# This used the CURRENT target label.
#
# NEW VERSION:
#
# We calculate consecutive bullish weeks using ONLY PREVIOUS labels.
#
# Example:
#
# Previous labels:
#
# 1, 1, 1, 0, 1
#
# Current row sees:
#
# 1, 1, 1, 0, 1
#
# but never its own current target.
# =============================================================================


previous_label = df["label"].shift(1)

consecutive_up = np.zeros(len(df), dtype=float)

count = 0

for i in range(len(df)):

    if pd.isna(previous_label.iloc[i]):
        count = 0
        consecutive_up[i] = 0

    elif previous_label.iloc[i] == 1:
        count += 1
        consecutive_up[i] = count

    else:
        count = 0
        consecutive_up[i] = 0


df["consec_up"] = consecutive_up


# =============================================================================
# OBV MOMENTUM
# =============================================================================

df["obv_momentum"] = (
    df["obv"].pct_change(4)
)


# =============================================================================
# VOLATILITY REGIME
# =============================================================================

df["vol_20w"] = (
    df["return_1w"].rolling(20).std()
)

df["vol_ratio"] = (
    df["return_1w"].rolling(4).std() /
    (df["vol_20w"] + 1e-9)
)


# =============================================================================
# RSI DIVERGENCE PROXY
# =============================================================================

df["price_up_rsi_down"] = (
    (df["Close"] > df["Close"].shift(1)) &
    (df["rsi_14"] < df["rsi_14"].shift(1))
).astype(int)


# =============================================================================
# REMOVE INITIAL NaN VALUES
# =============================================================================

df.dropna(inplace=True)


# =============================================================================
# FEATURE SELECTION
# =============================================================================
#
# The current target "label" is explicitly removed.
#
# OHLCV is also removed because the engineered indicators/price features
# already represent the information we want to use.
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


X = df[feature_cols]
y = df["label"]


# =============================================================================
# NUMERICAL SAFETY
# =============================================================================

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


print(f"\nTotal features: {X.shape[1]}")


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
#
# Chronological split.
#
# Training:
#     Before 2024
#
# Testing:
#     2024 onward
# =============================================================================

X_train = X[
    X.index < SPLIT_DATE
]

X_test = X[
    X.index >= SPLIT_DATE
]

y_train = y[
    y.index < SPLIT_DATE
]

y_test = y[
    y.index >= SPLIT_DATE
]


print(
    f"\nTrain: {X_train.shape[0]} weeks "
    f"({X_train.index[0].date()} → "
    f"{X_train.index[-1].date()})"
)

print(
    f"Test : {X_test.shape[0]} weeks  "
    f"({X_test.index[0].date()} → "
    f"{X_test.index[-1].date()})"
)

print(
    f"Train label balance: "
    f"{y_train.value_counts().to_dict()}"
)

print(
    f"Test  label balance: "
    f"{y_test.value_counts().to_dict()}"
)


# =============================================================================
# MODEL 1 — RANDOM FOREST
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


print(
    f"\nRF OOB Score: "
    f"{rf.oob_score_:.4f}"
)


# =============================================================================
# MODEL 2 — GRADIENT BOOSTING
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
# MODEL 3 — SOFT VOTING ENSEMBLE
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

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)


for name, model in [
    ("Random Forest", rf),
    ("Gradient Boosting", gbc),
    ("Ensemble", ensemble)
]:

    y_pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    print(f"\n{name}")

    print(
        f"  Accuracy : {acc:.4f}"
    )

    print(
        f"  Confusion Matrix:\n"
        f"{confusion_matrix(y_test, y_pred)}"
    )


# =============================================================================
# DETAILED REPORT — ENSEMBLE
# =============================================================================

print("\n" + "=" * 60)
print("  DETAILED REPORT — ENSEMBLE MODEL")
print("=" * 60)


y_pred = ensemble.predict(
    X_test
)

y_proba = ensemble.predict_proba(
    X_test
)[:, 1]


print(
    f"\nAccuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)


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


# =============================================================================
# PROBABILITY RESULTS
# =============================================================================

df_results = X_test.copy()

df_results["actual"] = (
    y_test.values
)

df_results["prob_up"] = (
    y_proba
)

df_results["predicted"] = (
    y_pred
)


print("\nSample probabilities:")

print(
    df_results[
        [
            "prob_up",
            "predicted",
            "actual"
        ]
    ].head(10)
)


# =============================================================================
# HIGH-CONFIDENCE FILTER
# =============================================================================

hc = df_results[
    df_results["prob_up"] >= PROB_THRESHOLD
]


if len(hc) > 0:

    hc_acc = accuracy_score(
        hc["actual"],
        hc["predicted"]
    )

    print(
        f"\nHigh-confidence weeks "
        f"(prob >= {PROB_THRESHOLD}): "
        f"{len(hc)}"
    )

    print(
        f"High-confidence accuracy: "
        f"{hc_acc:.4f}"
    )

    print(
        f"High-confidence label balance: "
        f"{hc['actual'].value_counts().to_dict()}"
    )

else:

    print(
        f"\nNo high-confidence weeks at "
        f"threshold {PROB_THRESHOLD}."
    )


# =============================================================================
# MULTIPLE CONFIDENCE THRESHOLDS
# =============================================================================

print("\nConfidence threshold analysis:")


for thresh in [
    0.55,
    0.60,
    0.65,
    0.70
]:

    hc_t = df_results[
        df_results["prob_up"] >= thresh
    ]

    if len(hc_t) > 0:

        acc_t = accuracy_score(
            hc_t["actual"],
            hc_t["predicted"]
        )

        print(
            f"  Threshold {thresh}: "
            f"{len(hc_t)} weeks | "
            f"accuracy {acc_t:.4f}"
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


print("\nTop 20 Features (RF):")

print(
    importance.head(20).round(4)
)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================
#
# TimeSeriesSplit ensures that future observations are never used to train
# a model before the corresponding validation period.
# =============================================================================

print("\n" + "=" * 60)
print("  WALK-FORWARD VALIDATION (5 folds)")
print("=" * 60)


tscv = TimeSeriesSplit(
    n_splits=5
)

wf_acc = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    tscv.split(X)
):

    Xtr = X.iloc[
        train_idx
    ]

    Xte = X.iloc[
        test_idx
    ]

    ytr = y.iloc[
        train_idx
    ]

    yte = y.iloc[
        test_idx
    ]


    fold_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
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
        f"  Fold {fold + 1}: "
        f"{len(Xtr)} train | "
        f"{len(Xte)} test | "
        f"accuracy {fold_acc:.4f}"
    )


print(
    f"\n  Mean accuracy : "
    f"{np.mean(wf_acc):.4f}"
)

print(
    f"  Std           : "
    f"{np.std(wf_acc):.4f}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

rf_acc = accuracy_score(
    y_test,
    rf.predict(X_test)
)

gbc_acc = accuracy_score(
    y_test,
    gbc.predict(X_test)
)

ensemble_acc = accuracy_score(
    y_test,
    ensemble.predict(X_test)
)


print("\n" + "=" * 60)
print(
    f"  FINAL SUMMARY — {INDEX_NAME}"
)
print("=" * 60)

print(
    f"  Features used        : "
    f"{X.shape[1]}"
)

print(
    f"  Train size           : "
    f"{X_train.shape[0]} weeks"
)

print(
    f"  Test size            : "
    f"{X_test.shape[0]} weeks"
)

print(
    f"  RF Accuracy          : "
    f"{rf_acc:.4f}"
)

print(
    f"  GBC Accuracy         : "
    f"{gbc_acc:.4f}"
)

print(
    f"  Ensemble Accuracy    : "
    f"{ensemble_acc:.4f}"
)

print(
    f"  Walk-forward mean    : "
    f"{np.mean(wf_acc):.4f} "
    f"± {np.std(wf_acc):.4f}"
)

print(
    f"  OOB Score (RF)       : "
    f"{rf.oob_score_:.4f}"
)

print("=" * 60)

