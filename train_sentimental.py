
import os
import pickle
import warnings

import numpy as np
import pandas as pd

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

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

# The script should be in the same folder as your CSV files.
PROJECT_PATH = os.path.dirname(
    os.path.abspath(__file__)
)

# Select:
# NIFTY / BANKNIFTY / SENSEX
INDEX_NAME = "SENSEX"

SPLIT_DATE = "2024-01-01"

PROB_THRESHOLD = 0.60


MARKET_FILES = {
    "NIFTY": "nifty_weekly_labeled.csv",
    "BANKNIFTY": "banknifty_weekly_labeled.csv",
    "SENSEX": "sensex_weekly_labeled.csv"
}


GDELT_FILE = "gdelt_weekly_sentiment.pkl"


MARKET_FILE = os.path.join(
    PROJECT_PATH,
    MARKET_FILES[INDEX_NAME]
)

GDELT_PATH = os.path.join(
    PROJECT_PATH,
    GDELT_FILE
)


# =============================================================================
# LOAD MARKET DATA
# =============================================================================

print("=" * 70)
print("        GDELT SENTIMENT-ONLY WEEKLY MODEL")
print("=" * 70)

print("\nLoading market data...")

market = pd.read_csv(
    MARKET_FILE,
    parse_dates=["Date"]
)

market = market.sort_values("Date")

market.set_index(
    "Date",
    inplace=True
)


print(
    f"Loaded {INDEX_NAME}: "
    f"{len(market)} rows"
)

print(
    f"Market date range: "
    f"{market.index.min().date()} → "
    f"{market.index.max().date()}"
)


# =============================================================================
# LOAD WEEKLY GDELT SENTIMENT
# =============================================================================

print("\nLoading GDELT weekly sentiment...")

with open(
    GDELT_PATH,
    "rb"
) as f:

    gdelt = pickle.load(f)


if not isinstance(
    gdelt,
    pd.DataFrame
):

    raise TypeError(
        "gdelt_weekly_sentiment.pkl "
        "does not contain a pandas DataFrame."
    )


print(
    f"GDELT rows: "
    f"{len(gdelt)}"
)

print(
    f"GDELT columns: "
    f"{list(gdelt.columns)}"
)


# =============================================================================
# VERIFY REQUIRED COLUMNS
# =============================================================================

REQUIRED_GDELT_COLUMNS = [
    "week",
    "mean_sentiment",
    "median_sentiment",
    "sentiment_std",
    "news_count"
]


missing_columns = [
    col
    for col in REQUIRED_GDELT_COLUMNS
    if col not in gdelt.columns
]


if missing_columns:

    raise ValueError(
        "Missing GDELT columns: "
        + str(missing_columns)
    )


# =============================================================================
# CONVERT GDELT WEEK RANGE TO DATE
# =============================================================================
#
# Example:
#
# 2015-02-14/2015-02-20
#
# becomes:
#
# 2015-02-20
#
# We use the END of the GDELT week as the weekly timestamp.
# =============================================================================

gdelt["week_start"] = pd.to_datetime(
    gdelt["week"].str.split("/").str[0],
    errors="coerce"
)

gdelt["week_end"] = pd.to_datetime(
    gdelt["week"].str.split("/").str[1],
    errors="coerce"
)


gdelt.dropna(
    subset=["week_end"],
    inplace=True
)


# =============================================================================
# SORT GDELT DATA
# =============================================================================

gdelt.sort_values(
    "week_end",
    inplace=True
)


gdelt.set_index(
    "week_end",
    inplace=True
)


# =============================================================================
# SELECT SENTIMENT FEATURES
# =============================================================================

SENTIMENT_FEATURES = [
    "mean_sentiment",
    "median_sentiment",
    "sentiment_std",
    "news_count"
]


sentiment = gdelt[
    SENTIMENT_FEATURES
].copy()


# =============================================================================
# CHECK DUPLICATE WEEKS
# =============================================================================

if sentiment.index.duplicated().any():

    print(
        "\nWARNING: Duplicate GDELT weeks detected."
    )

    sentiment = (
        sentiment
        .groupby(level=0)
        .mean()
    )


# =============================================================================
# DISPLAY GDELT INFORMATION
# =============================================================================

print(
    "\nGDELT weekly date range: "
    f"{sentiment.index.min().date()} → "
    f"{sentiment.index.max().date()}"
)


print(
    "\nGDELT missing values:"
)

print(
    sentiment.isna().sum()
)


print(
    "\nGDELT sample:"
)

print(
    sentiment.head()
)


# =============================================================================
# ALIGN MARKET DATA WITH GDELT
# =============================================================================
#
# IMPORTANT:
#
# We do NOT forward-fill sentiment across months where there is no GDELT
# information.
#
# We only keep weeks for which a real GDELT sentiment record exists.
# =============================================================================

combined = market[
    ["label"]
].join(
    sentiment,
    how="inner"
)


combined.sort_index(
    inplace=True
)


print(
    "\n" + "=" * 70
)

print(
    "AFTER MARKET + GDELT ALIGNMENT"
)

print(
    "=" * 70
)

print(
    f"Overlapping weeks: "
    f"{len(combined)}"
)

print(
    f"Overlap date range: "
    f"{combined.index.min().date()} → "
    f"{combined.index.max().date()}"
)


# =============================================================================
# REMOVE MISSING VALUES
# =============================================================================

combined.dropna(
    subset=[
        "label"
    ] + SENTIMENT_FEATURES,
    inplace=True
)


# =============================================================================
# DISPLAY LABEL BALANCE
# =============================================================================

print(
    "\nCombined label balance:"
)

print(
    combined["label"]
    .value_counts()
    .to_dict()
)


# =============================================================================
# FEATURE MATRIX
# =============================================================================

X = combined[
    SENTIMENT_FEATURES
].copy()

y = combined[
    "label"
].copy()


# =============================================================================
# NUMERICAL SAFETY
# =============================================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.astype(
    "float64"
)


# =============================================================================
# OPTIONAL SENTIMENT-DERIVED FEATURES
# =============================================================================
#
# We keep the original four GDELT variables.
#
# These additional features are based only on CURRENT/PAST sentiment
# information and do not use the target.
# =============================================================================

X["sentiment_change"] = (
    X["mean_sentiment"].diff()
)

X["sentiment_2w_change"] = (
    X["mean_sentiment"].diff(2)
)

X["sentiment_4w_avg"] = (
    X["mean_sentiment"].rolling(4).mean()
)

X["sentiment_4w_std"] = (
    X["mean_sentiment"].rolling(4).std()
)

X["news_count_change"] = (
    X["news_count"].pct_change()
)

X["news_count_4w_avg"] = (
    X["news_count"].rolling(4).mean()
)


# =============================================================================
# REMOVE INITIAL NaN VALUES CREATED BY ROLLING FEATURES
# =============================================================================

valid = X.notna().all(axis=1)

X = X.loc[valid]

y = y.loc[X.index]


# =============================================================================
# REPLACE ANY REMAINING INVALID VALUES
# =============================================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.clip(
    lower=-1e6,
    upper=1e6
)


print(
    f"\nTotal sentiment features: "
    f"{X.shape[1]}"
)

print(
    "Features:"
)

for feature in X.columns:
    print(
        f"  - {feature}"
    )


# =============================================================================
# TRAIN / TEST SPLIT
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
    "\n" + "=" * 70
)

print(
    "TRAIN / TEST SPLIT"
)

print(
    "=" * 70
)


print(
    f"Train: "
    f"{len(X_train)} weeks "
    f"({X_train.index.min().date()} → "
    f"{X_train.index.max().date()})"
)

print(
    f"Test : "
    f"{len(X_test)} weeks "
    f"({X_test.index.min().date()} → "
    f"{X_test.index.max().date()})"
)


print(
    "\nTrain label balance:"
)

print(
    y_train.value_counts().to_dict()
)


print(
    "\nTest label balance:"
)

print(
    y_test.value_counts().to_dict()
)


# =============================================================================
# MODEL 1 — RANDOM FOREST
# =============================================================================

print(
    "\nTraining Random Forest..."
)


rf = RandomForestClassifier(
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


rf.fit(
    X_train,
    y_train
)


print(
    f"RF OOB Score: "
    f"{rf.oob_score_:.4f}"
)


# =============================================================================
# MODEL 2 — GRADIENT BOOSTING
# =============================================================================

print(
    "Training Gradient Boosting..."
)


gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=3,
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
# MODEL 3 — SOFT VOTING
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

print(
    "\n" + "=" * 70
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 70
)


models = [
    ("Random Forest", rf),
    ("Gradient Boosting", gbc),
    ("Ensemble", ensemble)
]


for name, model in models:

    pred = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Accuracy : "
        f"{acc:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )


# =============================================================================
# DETAILED ENSEMBLE REPORT
# =============================================================================

ensemble_pred = ensemble.predict(
    X_test
)

ensemble_prob = ensemble.predict_proba(
    X_test
)[:, 1]


print(
    "\n" + "=" * 70
)

print(
    "DETAILED REPORT — SENTIMENT ENSEMBLE"
)

print(
    "=" * 70
)


print(
    f"\nAccuracy: "
    f"{accuracy_score(y_test, ensemble_pred):.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        ensemble_pred,
        target_names=[
            "Bearish",
            "Bullish"
        ]
    )
)


# =============================================================================
# CONFIDENCE ANALYSIS
# =============================================================================

results = pd.DataFrame(
    index=X_test.index
)

results["prob_up"] = (
    ensemble_prob
)

results["predicted"] = (
    ensemble_pred
)

results["actual"] = (
    y_test
)


print(
    "\nSample probabilities:"
)

print(
    results.head(10)
)


print(
    "\nConfidence threshold analysis:"
)


for threshold in [
    0.55,
    0.60,
    0.65,
    0.70
]:

    selected = results[
        results["prob_up"] >= threshold
    ]

    if len(selected) == 0:
        continue

    accuracy = accuracy_score(
        selected["actual"],
        selected["predicted"]
    )

    print(
        f"  Threshold {threshold}: "
        f"{len(selected)} weeks | "
        f"accuracy {accuracy:.4f}"
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print(
    "\n" + "=" * 70
)

print(
    "SENTIMENT FEATURE IMPORTANCE"
)

print(
    "=" * 70
)


importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


print(
    importance.round(4)
)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

print(
    "\n" + "=" * 70
)

print(
    "WALK-FORWARD VALIDATION"
)

print(
    "=" * 70
)


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
        max_depth=5,
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


    fold_accuracy = accuracy_score(
        yte,
        fold_pred
    )


    wf_acc.append(
        fold_accuracy
    )


    print(
        f"  Fold {fold + 1}: "
        f"{len(Xtr)} train | "
        f"{len(Xte)} test | "
        f"accuracy {fold_accuracy:.4f}"
    )


print(
    f"\nMean accuracy : "
    f"{np.mean(wf_acc):.4f}"
)

print(
    f"Std           : "
    f"{np.std(wf_acc):.4f}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

rf_accuracy = accuracy_score(
    y_test,
    rf.predict(X_test)
)

gbc_accuracy = accuracy_score(
    y_test,
    gbc.predict(X_test)
)

ensemble_accuracy = accuracy_score(
    y_test,
    ensemble.predict(X_test)
)


print(
    "\n" + "=" * 70
)

print(
    f"FINAL SUMMARY — GDELT SENTIMENT — {INDEX_NAME}"
)

print(
    "=" * 70
)

print(
    f"Sentiment features : "
    f"{X.shape[1]}"
)

print(
    f"Train size         : "
    f"{len(X_train)} weeks"
)

print(
    f"Test size          : "
    f"{len(X_test)} weeks"
)

print(
    f"RF Accuracy        : "
    f"{rf_accuracy:.4f}"
)

print(
    f"GBC Accuracy       : "
    f"{gbc_accuracy:.4f}"
)

print(
    f"Ensemble Accuracy  : "
    f"{ensemble_accuracy:.4f}"
)

print(
    f"Walk-forward mean  : "
    f"{np.mean(wf_acc):.4f} "
    f"± {np.std(wf_acc):.4f}"
)

print(
    f"RF OOB Score       : "
    f"{rf.oob_score_:.4f}"
)

print(
    "=" * 70
)

