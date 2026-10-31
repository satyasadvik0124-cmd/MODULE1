
"""
======================================================================
        HYBRID TECHNICAL + GDELT WEEKLY MODEL
======================================================================

Combines:
    1. Leakage-free technical ML probability
    2. GDELT sentiment ML probability

Fusion:
    P(Hybrid) = Technical_Weight * P(Technical)
              + Sentiment_Weight * P(Sentiment)

Tests:
    100/0
    90/10
    80/20
    70/30
    60/40
    50/50

Run separately for:
    NIFTY
    BANKNIFTY
    SENSEX

IMPORTANT:
    This script does NOT use the old "consec_up" target-leaking
    implementation.

    GDELT ends at 2026-01-23, so only common dates are used.
======================================================================
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Change this:
INDEX_NAME = "NIFTY"

# Fixed chronological split
SPLIT_DATE = "2024-01-01"

# Fusion weights
FUSION_WEIGHTS = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
]


# ======================================================================
# FILE PATHS
# ======================================================================

MARKET_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_weekly_labeled.csv"
)

GDELT_FILE = os.path.join(
    PROJECT_PATH,
    "gdelt_weekly_sentiment.pkl"
)


# ======================================================================
# TECHNICAL FEATURE ENGINEERING
# ======================================================================

def create_technical_features(df):
    """
    Create leakage-free technical features.

    IMPORTANT:
    The target label is NEVER used directly to construct
    a feature for the same row.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Basic price features
    # --------------------------------------------------------------

    if all(col in df.columns for col in ["High", "Low", "Close"]):
        df["price_range"] = (
            (df["High"] - df["Low"]) /
            df["Close"].replace(0, np.nan)
        )

        df["close_position"] = (
            (df["Close"] - df["Low"]) /
            (df["High"] - df["Low"]).replace(0, np.nan)
        )

    # --------------------------------------------------------------
    # Gap
    # --------------------------------------------------------------

    if "Open" in df.columns:
        df["gap"] = (
            (df["Open"] - df["Close"].shift(1)) /
            df["Close"].shift(1).replace(0, np.nan)
        )

    # --------------------------------------------------------------
    # EMA slopes
    # --------------------------------------------------------------

    if "EMA_20" in df.columns:
        df["ema_20_slope"] = df["EMA_20"].pct_change()

    if "EMA_50" in df.columns:
        df["ema_50_slope"] = df["EMA_50"].pct_change()

    # --------------------------------------------------------------
    # EMA cross features
    # --------------------------------------------------------------

    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        df["ema_20_50_cross"] = (
            df["EMA_20"] > df["EMA_50"]
        ).astype(int)

    if "EMA_50" in df.columns and "EMA_100" in df.columns:
        df["ema_50_100_cross"] = (
            df["EMA_50"] > df["EMA_100"]
        ).astype(int)

    # --------------------------------------------------------------
    # RSI
    # --------------------------------------------------------------

    if "RSI" in df.columns:
        df["rsi_overbought"] = (
            df["RSI"] > 70
        ).astype(int)

        df["rsi_oversold"] = (
            df["RSI"] < 30
        ).astype(int)

        df["rsi_mid"] = (
            (df["RSI"] >= 40) &
            (df["RSI"] <= 60)
        ).astype(int)

        df["rsi_change"] = df["RSI"].diff()

    # --------------------------------------------------------------
    # MACD
    # --------------------------------------------------------------

    if "MACD" in df.columns and "MACD_signal" in df.columns:
        macd_diff = df["MACD"] - df["MACD_signal"]

        df["macd_cross"] = (
            macd_diff > 0
        ).astype(int)

        df["macd_cross_change"] = macd_diff.diff()

    # --------------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------------

    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        df["bb_width"] = (
            (df["BB_upper"] - df["BB_lower"]) /
            df["Close"].replace(0, np.nan)
        )

        df["bb_squeeze"] = (
            df["bb_width"] <
            df["bb_width"].rolling(20).quantile(0.2)
        ).astype(int)

    # --------------------------------------------------------------
    # ADX / trend
    # --------------------------------------------------------------

    if "ADX" in df.columns:
        df["trend_strong"] = (
            df["ADX"] > 25
        ).astype(int)

    if "DI_plus" in df.columns and "DI_minus" in df.columns:
        df["di_cross"] = (
            df["DI_plus"] > df["DI_minus"]
        ).astype(int)

    # --------------------------------------------------------------
    # Returns
    # --------------------------------------------------------------

    if "Close" in df.columns:
        for period in [1, 2, 4, 8, 12]:
            df[f"return_{period}w"] = (
                df["Close"].pct_change(period)
            )

        df["mom_accel"] = (
            df["return_1w"] -
            df["return_4w"]
        )

    # --------------------------------------------------------------
    # Previous labels ONLY
    #
    # These are valid because they contain information available
    # before the current prediction.
    # --------------------------------------------------------------

    if "label" in df.columns:

        df["label_lag1"] = df["label"].shift(1)
        df["label_lag2"] = df["label"].shift(2)
        df["label_lag3"] = df["label"].shift(3)

        previous_label = df["label"].shift(1)

        df["consec_up"] = (
            previous_label
            .groupby(
                (previous_label != previous_label.shift()).cumsum()
            )
            .cumcount()
        )

        df["consec_up"] = (
            df["consec_up"] * previous_label
        )

    # --------------------------------------------------------------
    # OBV momentum
    # --------------------------------------------------------------

    if "OBV" in df.columns:
        df["obv_momentum"] = df["OBV"].pct_change()

    # --------------------------------------------------------------
    # Volatility
    # --------------------------------------------------------------

    if "return_1w" in df.columns:
        df["vol_20w"] = (
            df["return_1w"]
            .rolling(20)
            .std()
        )

    if "Volume" in df.columns:
        volume_mean = (
            df["Volume"]
            .rolling(20)
            .mean()
        )

        df["vol_ratio"] = (
            df["Volume"] /
            volume_mean.replace(0, np.nan)
        )

    # --------------------------------------------------------------
    # Price / RSI divergence
    # --------------------------------------------------------------

    if "return_1w" in df.columns and "RSI" in df.columns:

        price_up = df["return_1w"] > 0
        rsi_down = df["RSI"].diff() < 0

        df["price_up_rsi_down"] = (
            price_up & rsi_down
        ).astype(int)

    return df


# ======================================================================
# LOAD MARKET DATA
# ======================================================================

print("\n" + "=" * 70)
print("        HYBRID TECHNICAL + GDELT WEEKLY MODEL")
print("=" * 70)

print("\nLoading market data...")

if not os.path.exists(MARKET_FILE):
    raise FileNotFoundError(
        f"Market file not found:\n{MARKET_FILE}"
    )

market = pd.read_csv(MARKET_FILE)

print(f"Loaded {INDEX_NAME}: {len(market)} rows")


# ======================================================================
# DATE HANDLING
# ======================================================================

date_candidates = [
    "Date",
    "date",
    "Datetime",
    "datetime"
]

date_column = None

for col in date_candidates:
    if col in market.columns:
        date_column = col
        break

if date_column is None:
    raise ValueError(
        "Could not find Date/date/Datetime column."
    )

market[date_column] = pd.to_datetime(
    market[date_column]
)

market = market.sort_values(
    date_column
).set_index(date_column)

print(
    f"Market date range: "
    f"{market.index.min().date()} → "
    f"{market.index.max().date()}"
)


# ======================================================================
# CHECK LABEL
# ======================================================================

if "label" not in market.columns:
    raise ValueError(
        "Market dataset must contain 'label'."
    )


# ======================================================================
# CREATE TECHNICAL FEATURES
# ======================================================================

print("\nCreating leakage-free technical features...")

market = create_technical_features(market)


# ======================================================================
# LOAD GDELT
# ======================================================================

print("\nLoading GDELT weekly sentiment...")

if not os.path.exists(GDELT_FILE):
    raise FileNotFoundError(
        f"GDELT file not found:\n{GDELT_FILE}"
    )

with open(GDELT_FILE, "rb") as f:
    gdelt = pickle.load(f)

print(f"GDELT rows: {len(gdelt)}")
print(f"GDELT columns: {list(gdelt.columns)}")


# ======================================================================
# PARSE GDELT WEEK
# ======================================================================

gdelt["week"] = gdelt["week"].astype(str)

week_parts = gdelt["week"].str.split("/", expand=True)

gdelt["week_start"] = pd.to_datetime(
    week_parts[0]
)

gdelt["week_end"] = pd.to_datetime(
    week_parts[1]
)

gdelt = gdelt.set_index(
    "week_end"
).sort_index()

print(
    f"GDELT weekly date range: "
    f"{gdelt.index.min().date()} → "
    f"{gdelt.index.max().date()}"
)


# ======================================================================
# GDELT FEATURES
# ======================================================================

sentiment_features = [
    "mean_sentiment",
    "median_sentiment",
    "sentiment_std",
    "news_count"
]

missing_sentiment = [
    col for col in sentiment_features
    if col not in gdelt.columns
]

if missing_sentiment:
    raise ValueError(
        f"Missing GDELT columns: {missing_sentiment}"
    )


# ======================================================================
# DERIVED SENTIMENT FEATURES
# ======================================================================

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
    gdelt["news_count"].diff()
)

gdelt["news_count_4w_avg"] = (
    gdelt["news_count"]
    .rolling(4)
    .mean()
)


sentiment_features = [
    "mean_sentiment",
    "median_sentiment",
    "sentiment_std",
    "news_count",
    "sentiment_change",
    "sentiment_2w_change",
    "sentiment_4w_avg",
    "sentiment_4w_std",
    "news_count_change",
    "news_count_4w_avg"
]


# ======================================================================
# ALIGN MARKET + GDELT
# ======================================================================

print("\n" + "=" * 70)
print("AFTER MARKET + GDELT ALIGNMENT")
print("=" * 70)

market.index = pd.to_datetime(
    market.index
)

gdelt.index = pd.to_datetime(
    gdelt.index
)

common_dates = market.index.intersection(
    gdelt.index
)

print(f"Overlapping weeks: {len(common_dates)}")

print(
    f"Overlap date range: "
    f"{common_dates.min().date()} → "
    f"{common_dates.max().date()}"
)


# ======================================================================
# CREATE COMMON DATASET
# ======================================================================

combined = market.loc[
    common_dates
].copy()

sentiment_aligned = gdelt.loc[
    common_dates,
    sentiment_features
].copy()

combined = combined.join(
    sentiment_aligned,
    how="inner"
)


# ======================================================================
# REMOVE NaNs
# ======================================================================

combined = combined.dropna(
    subset=sentiment_features
)


print(
    f"Final common dataset: "
    f"{len(combined)} weeks"
)

print(
    f"Date range after feature creation: "
    f"{combined.index.min().date()} → "
    f"{combined.index.max().date()}"
)


# ======================================================================
# TECHNICAL FEATURES
# ======================================================================

technical_exclude = {
    "label",
    "future_close"
}

technical_features = [
    col
    for col in combined.columns
    if col not in sentiment_features
    and col not in technical_exclude
    and pd.api.types.is_numeric_dtype(
        combined[col]
    )
]


# Remove columns that are completely NaN
technical_features = [
    col for col in technical_features
    if combined[col].notna().any()
]


# Drop rows missing technical features
combined = combined.dropna(
    subset=technical_features + sentiment_features
)


print(
    f"\nTechnical features: "
    f"{len(technical_features)}"
)

for feature in technical_features:
    print(f"  - {feature}")

print(
    f"\nSentiment features: "
    f"{len(sentiment_features)}"
)

for feature in sentiment_features:
    print(f"  - {feature}")


# ======================================================================
# FINAL COMMON DATASET
# ======================================================================

X_technical = combined[
    technical_features
]

X_sentiment = combined[
    sentiment_features
]

y = combined[
    "label"
].astype(int)


# ======================================================================
# TRAIN / TEST SPLIT
# ======================================================================

train_mask = (
    combined.index < SPLIT_DATE
)

test_mask = (
    combined.index >= SPLIT_DATE
)

X_tech_train = X_technical.loc[
    train_mask
]

X_tech_test = X_technical.loc[
    test_mask
]

X_sent_train = X_sentiment.loc[
    train_mask
]

X_sent_test = X_sentiment.loc[
    test_mask
]

y_train = y.loc[
    train_mask
]

y_test = y.loc[
    test_mask
]


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Train: {len(y_train)} weeks "
    f"({y_train.index.min().date()} → "
    f"{y_train.index.max().date()})"
)

print(
    f"Test : {len(y_test)} weeks "
    f"({y_test.index.min().date()} → "
    f"{y_test.index.max().date()})"
)

print(
    "\nTrain label balance:"
)

print(
    y_train.value_counts().sort_index().to_dict()
)

print(
    "\nTest label balance:"
)

print(
    y_test.value_counts().sort_index().to_dict()
)


# ======================================================================
# TECHNICAL MODEL
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING TECHNICAL MODEL")
print("=" * 70)

technical_rf = RandomForestClassifier(
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

technical_rf.fit(
    X_tech_train,
    y_train
)

tech_probability = (
    technical_rf
    .predict_proba(X_tech_test)[:, 1]
)

tech_prediction = (
    tech_probability >= 0.50
).astype(int)

print(
    f"Technical RF OOB: "
    f"{technical_rf.oob_score_:.4f}"
)


# ======================================================================
# SENTIMENT MODEL
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING GDELT SENTIMENT MODEL")
print("=" * 70)

sentiment_rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

sentiment_rf.fit(
    X_sent_train,
    y_train
)

sent_probability = (
    sentiment_rf
    .predict_proba(X_sent_test)[:, 1]
)

sent_prediction = (
    sent_probability >= 0.50
).astype(int)

print(
    f"Sentiment RF OOB: "
    f"{sentiment_rf.oob_score_:.4f}"
)


# ======================================================================
# BASELINE RESULTS
# ======================================================================

technical_accuracy = accuracy_score(
    y_test,
    tech_prediction
)

sentiment_accuracy = accuracy_score(
    y_test,
    sent_prediction
)

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    f"Technical RF Accuracy : "
    f"{technical_accuracy:.4f}"
)

print(
    f"GDELT RF Accuracy     : "
    f"{sentiment_accuracy:.4f}"
)


# ======================================================================
# HYBRID FUSION
# ======================================================================

results = []

print("\n" + "=" * 70)
print("HYBRID PROBABILITY FUSION")
print("=" * 70)

for tech_weight, sent_weight in FUSION_WEIGHTS:

    hybrid_probability = (
        tech_weight * tech_probability
        +
        sent_weight * sent_probability
    )

    hybrid_prediction = (
        hybrid_probability >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        hybrid_prediction
    )

    precision = precision_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    results.append({
        "technical_weight": tech_weight,
        "sentiment_weight": sent_weight,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"\nTechnical {tech_weight:.0%} "
        f"+ GDELT {sent_weight:.0%}"
    )

    print(
        f"  Accuracy  : {accuracy:.4f}"
    )

    print(
        f"  Precision : {precision:.4f}"
    )

    print(
        f"  Recall    : {recall:.4f}"
    )

    print(
        f"  F1        : {f1:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            hybrid_prediction
        )
    )


# ======================================================================
# RESULTS TABLE
# ======================================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("HYBRID RESULTS SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ======================================================================
# BEST ACCURACY CONFIGURATION
# ======================================================================

best_index = (
    results_df["accuracy"]
    .idxmax()
)

best_result = (
    results_df.loc[best_index]
)

print("\n" + "=" * 70)
print("BEST HYBRID CONFIGURATION")
print("=" * 70)

print(
    f"Technical weight : "
    f"{best_result['technical_weight']:.0%}"
)

print(
    f"GDELT weight     : "
    f"{best_result['sentiment_weight']:.0%}"
)

print(
    f"Accuracy         : "
    f"{best_result['accuracy']:.4f}"
)

print(
    f"Precision        : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall           : "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1               : "
    f"{best_result['f1']:.4f}"
)



# ======================================================================
# WALK-FORWARD HYBRID VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("WALK-FORWARD HYBRID VALIDATION")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)

walk_forward_results = []

for tech_weight, sent_weight in FUSION_WEIGHTS:

    fold_scores = []

    print(
        f"\nTechnical {tech_weight:.0%} + "
        f"GDELT {sent_weight:.0%}"
    )

    fold_number = 1

    for train_idx, val_idx in tscv.split(combined):

        # ==============================================================
        # GET TRAINING / VALIDATION DATA
        # ==============================================================

        Xtr_tech = X_technical.iloc[train_idx]
        Xval_tech = X_technical.iloc[val_idx]

        Xtr_sent = X_sentiment.iloc[train_idx]
        Xval_sent = X_sentiment.iloc[val_idx]

        ytr = y.iloc[train_idx]
        yval = y.iloc[val_idx]

        # ==============================================================
        # TECHNICAL RANDOM FOREST
        # ==============================================================

        fold_tech_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        fold_tech_model.fit(
            Xtr_tech,
            ytr
        )

        fold_tech_prob = (
            fold_tech_model
            .predict_proba(Xval_tech)[:, 1]
        )

        # ==============================================================
        # GDELT RANDOM FOREST
        # ==============================================================

        fold_sent_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        fold_sent_model.fit(
            Xtr_sent,
            ytr
        )

        fold_sent_prob = (
            fold_sent_model
            .predict_proba(Xval_sent)[:, 1]
        )

        # ==============================================================
        # HYBRID PROBABILITY FUSION
        # ==============================================================

        fold_hybrid_prob = (
            tech_weight * fold_tech_prob
            + sent_weight * fold_sent_prob
        )

        fold_prediction = (
            fold_hybrid_prob >= 0.50
        ).astype(int)

        # ==============================================================
        # FOLD ACCURACY
        # ==============================================================

        fold_accuracy = accuracy_score(
            yval,
            fold_prediction
        )

        fold_scores.append(
            fold_accuracy
        )

        print(
            f"  Fold {fold_number}: "
            f"{len(ytr)} train | "
            f"{len(yval)} test | "
            f"accuracy {fold_accuracy:.4f}"
        )

        fold_number += 1

    # ==================================================================
    # CALCULATE WALK-FORWARD STATISTICS
    # ==================================================================

    mean_score = np.mean(fold_scores)

    std_score = np.std(fold_scores)

    walk_forward_results.append(
        {
            "technical_weight": tech_weight,
            "sentiment_weight": sent_weight,
            "walk_forward_mean": mean_score,
            "walk_forward_std": std_score
        }
    )

    print(
        f"  Mean accuracy : {mean_score:.4f}"
    )

    print(
        f"  Std           : {std_score:.4f}"
    )


# ======================================================================
# WALK-FORWARD SUMMARY
# ======================================================================

walk_forward_df = pd.DataFrame(
    walk_forward_results
)

print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

print(
    walk_forward_df.to_string(
        index=False
    )
)


# ======================================================================
# MERGE FIXED TEST + WALK-FORWARD RESULTS
# ======================================================================

final_results = results_df.merge(
    walk_forward_df,
    on=[
        "technical_weight",
        "sentiment_weight"
    ],
    how="left"
)

print("\n" + "=" * 70)
print("FINAL HYBRID COMPARISON")
print("=" * 70)

print(
    final_results.to_string(
        index=False
    )
)


# ======================================================================
# SAVE RESULTS
# ======================================================================

output_file = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_hybrid_results.csv"
)

final_results.to_csv(
    output_file,
    index=False
)

print(
    f"\nResults saved to:"
)

print(
    output_file
)


# ======================================================================
# FINAL MESSAGE
# ======================================================================

print("\n" + "=" * 70)
print(
    f"HYBRID MODEL COMPLETE — {INDEX_NAME}"
)
print("=" * 70)

print(
    "\nTechnical and GDELT probabilities were "
    "combined using weighted probability fusion."
)

print(
    "All fusion weights were evaluated using "
    "the same aligned weekly dataset."
)

print("=" * 70)

