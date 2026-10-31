
# ================================================================
# HYBRID FINANCIAL INTELLIGENCE SYSTEM
# MODULE 1 — RISK PREDICTION
#
# V2:
# COMPARE THREE DIFFERENT FUTURE-RISK TARGETS
#
# TARGET A:
#   Future 4-week endpoint return
#
# TARGET B:
#   Future 4-week minimum CLOSE return
#
# TARGET C:
#   Future 4-week minimum LOW return
#
# Models:
#   1. Linear Regression
#   2. Random Forest Regressor
#   3. Gradient Boosting Regressor
#
# Validation:
#   Strict chronological TimeSeriesSplit
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit


# ================================================================
# CONFIGURATION
# ================================================================

INDEX_NAME = "BANKNIFTY"

PROJECT_PATH = (
    "/content/drive/MyDrive/"
    "market_ml_project-20260815T095833Z-1-001/"
    "market_ml_project"
)

INPUT_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_drawdown_risk_dataset_v1.csv"
)

RESULTS_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_risk_target_comparison_v2.csv"
)

PREDICTIONS_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_risk_target_predictions_v2.csv"
)

FEATURES_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_risk_features_v2.csv"
)


# ================================================================
# WALK-FORWARD SETTINGS
# ================================================================

N_SPLITS = 5


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("BANKNIFTY RISK REGRESSION — V2")
print("TARGET COMPARISON")
print("=" * 75)

print(
    "Index       :",
    INDEX_NAME
)

print(
    "Walk-forward:",
    N_SPLITS,
    "folds"
)

print()


# ================================================================
# CHECK INPUT
# ================================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"""
Input file not found:

{INPUT_FILE}

Run Step 1 first.
"""
    )

print("Input dataset found.")


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"]
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print()
print("=" * 75)
print("INPUT DATA")
print("=" * 75)

print(
    "Rows       :",
    len(df)
)

print(
    "Date range :",
    df["Date"].min().date(),
    "→",
    df["Date"].max().date()
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ================================================================
# NUMERIC CONVERSION
# ================================================================

for col in [
    "Open",
    "High",
    "Low",
    "Close"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


if "Volume" in df.columns:

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

else:

    df["Volume"] = 0.0


df = df.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
).copy()

df = df.reset_index(drop=True)


# ================================================================
# CREATE THREE TARGETS
# ================================================================
#
# At time t:
#
# TARGET A:
#   Close[t+4] / Close[t] - 1
#
# TARGET B:
#   min(Close[t+1:t+4]) / Close[t] - 1
#
# TARGET C:
#   min(Low[t+1:t+4]) / Close[t] - 1
#
# All targets use ONLY future observations.
# Features will use information available at time t.
# ================================================================

print()
print("=" * 75)
print("CREATING THREE RISK TARGETS")
print("=" * 75)


# ------------------------------------------------
# Target A — endpoint return
# ------------------------------------------------

df["target_endpoint_return"] = (
    df["Close"].shift(-4)
    / df["Close"]
    - 1
)


# ------------------------------------------------
# Target B — minimum future closing return
# ------------------------------------------------

future_close_values = pd.concat(
    [
        df["Close"].shift(-1),
        df["Close"].shift(-2),
        df["Close"].shift(-3),
        df["Close"].shift(-4)
    ],
    axis=1
)

df["future_4w_min_close"] = (
    future_close_values.min(
        axis=1
    )
)

df["target_min_close_return"] = (
    df["future_4w_min_close"]
    / df["Close"]
    - 1
)


# ------------------------------------------------
# Target C — minimum future low return
# ------------------------------------------------

future_low_values = pd.concat(
    [
        df["Low"].shift(-1),
        df["Low"].shift(-2),
        df["Low"].shift(-3),
        df["Low"].shift(-4)
    ],
    axis=1
)

df["future_4w_min_low"] = (
    future_low_values.min(
        axis=1
    )
)

df["target_min_low_return"] = (
    df["future_4w_min_low"]
    / df["Close"]
    - 1
)


# ================================================================
# TARGET NAMES
# ================================================================

TARGETS = {

    "Endpoint 4W Return":
        "target_endpoint_return",

    "Minimum 4W Close Return":
        "target_min_close_return",

    "Minimum 4W Low Return":
        "target_min_low_return"
}


# ================================================================
# TARGET STATISTICS
# ================================================================

print()

for target_name, target_column in TARGETS.items():

    valid_target = (
        df[target_column]
        .dropna()
    )

    print("-" * 75)

    print(
        target_name
    )

    print(
        f"Observations : {len(valid_target)}"
    )

    print(
        f"Mean         : "
        f"{valid_target.mean() * 100:.2f}%"
    )

    print(
        f"Median       : "
        f"{valid_target.median() * 100:.2f}%"
    )

    print(
        f"Std          : "
        f"{valid_target.std() * 100:.2f}%"
    )

    print(
        f"Minimum      : "
        f"{valid_target.min() * 100:.2f}%"
    )

    print(
        f"Maximum      : "
        f"{valid_target.max() * 100:.2f}%"
    )


# ================================================================
# FEATURE ENGINEERING
# ================================================================

print()
print("=" * 75)
print("FEATURE ENGINEERING")
print("=" * 75)


close = df["Close"]

high = df["High"]

low = df["Low"]

open_price = df["Open"]

volume = df["Volume"]


# ================================================================
# TREND
# ================================================================

print("Trend features...")


df["ema20"] = (
    close
    .ewm(
        span=20,
        adjust=False
    )
    .mean()
)

df["ema50"] = (
    close
    .ewm(
        span=50,
        adjust=False
    )
    .mean()
)

df["ema100"] = (
    close
    .ewm(
        span=100,
        adjust=False
    )
    .mean()
)


df["ema20_vs_ema50"] = (
    df["ema20"]
    / df["ema50"]
    - 1
)

df["ema50_vs_ema100"] = (
    df["ema50"]
    / df["ema100"]
    - 1
)

df["ema20_slope"] = (
    df["ema20"]
    .pct_change(2)
)

df["ema50_slope"] = (
    df["ema50"]
    .pct_change(4)
)


# ================================================================
# RSI
# ================================================================

print("RSI...")


delta = close.diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

avg_gain = (
    gain
    .ewm(
        alpha=1 / 14,
        adjust=False
    )
    .mean()
)

avg_loss = (
    loss
    .ewm(
        alpha=1 / 14,
        adjust=False
    )
    .mean()
)

rs = (
    avg_gain /
    avg_loss.replace(
        0,
        np.nan
    )
)

df["rsi14"] = (
    100 -
    100 /
    (1 + rs)
)

df["rsi_distance_50"] = (
    df["rsi14"] - 50
)


# ================================================================
# MACD
# ================================================================

print("MACD...")


ema12 = (
    close
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

ema26 = (
    close
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

df["macd"] = (
    ema12 -
    ema26
)

df["macd_signal"] = (
    df["macd"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)

df["macd_hist"] = (
    df["macd"] -
    df["macd_signal"]
)


# ================================================================
# ATR
# ================================================================

print("ATR...")


previous_close = (
    close.shift(1)
)

tr1 = (
    high - low
)

tr2 = (
    high - previous_close
).abs()

tr3 = (
    low - previous_close
).abs()

true_range = pd.concat(
    [
        tr1,
        tr2,
        tr3
    ],
    axis=1
).max(axis=1)

df["atr14"] = (
    true_range
    .rolling(
        14,
        min_periods=14
    )
    .mean()
)

df["atr14_pct"] = (
    df["atr14"]
    / close
)


# ================================================================
# BOLLINGER WIDTH
# ================================================================

print("Bollinger Bands...")


bb_middle = (
    close
    .rolling(
        20,
        min_periods=20
    )
    .mean()
)

bb_std = (
    close
    .rolling(
        20,
        min_periods=20
    )
    .std()
)

bb_upper = (
    bb_middle +
    2 * bb_std
)

bb_lower = (
    bb_middle -
    2 * bb_std
)

df["bb_width"] = (
    (bb_upper - bb_lower)
    / bb_middle
)


# ================================================================
# RETURNS
# ================================================================

print("Returns...")


df["return_1w"] = (
    close.pct_change(1)
)

df["return_2w"] = (
    close.pct_change(2)
)

df["return_4w"] = (
    close.pct_change(4)
)

df["return_8w"] = (
    close.pct_change(8)
)

df["return_12w"] = (
    close.pct_change(12)
)


# ================================================================
# VOLATILITY
# ================================================================

print("Volatility...")


weekly_returns = (
    close.pct_change()
)

df["realized_vol_4w"] = (
    weekly_returns
    .rolling(
        4,
        min_periods=4
    )
    .std()
)

df["realized_vol_8w"] = (
    weekly_returns
    .rolling(
        8,
        min_periods=8
    )
    .std()
)

df["realized_vol_20w"] = (
    weekly_returns
    .rolling(
        20,
        min_periods=20
    )
    .std()
)

df["vol_ratio"] = (
    df["realized_vol_4w"]
    / df["realized_vol_20w"]
)


# ================================================================
# HISTORICAL DRAWDOWN
# ================================================================

print("Historical drawdown...")


rolling_high_20 = (
    close
    .rolling(
        20,
        min_periods=20
    )
    .max()
)

rolling_high_52 = (
    close
    .rolling(
        52,
        min_periods=52
    )
    .max()
)

df["drawdown_20w"] = (
    close /
    rolling_high_20
    - 1
)

df["drawdown_52w"] = (
    close /
    rolling_high_52
    - 1
)


# ================================================================
# OBV
# ================================================================

print("OBV...")


direction = np.sign(
    close.diff()
)

obv_change = (
    direction *
    volume
)

df["obv"] = (
    obv_change
    .fillna(0)
    .cumsum()
)

obv_previous = (
    df["obv"].shift(4)
)

df["obv_slope"] = (
    (
        df["obv"] -
        obv_previous
    )
    /
    (
        obv_previous.abs()
        + 1e-9
    )
)


# ================================================================
# PRICE STRUCTURE
# ================================================================

print("Price structure...")


df["range_pct"] = (
    high -
    low
) / close

df["body_pct"] = (
    (
        close -
        open_price
    ).abs()
    / close
)

df["close_position"] = (
    (close - low)
    /
    (
        high -
        low +
        1e-9
    )
)


# ================================================================
# FEATURE LIST
# ================================================================

FEATURES = [

    "ema20_vs_ema50",
    "ema50_vs_ema100",
    "ema20_slope",
    "ema50_slope",

    "rsi14",
    "rsi_distance_50",

    "macd",
    "macd_signal",
    "macd_hist",

    "atr14_pct",
    "bb_width",

    "realized_vol_4w",
    "realized_vol_8w",
    "realized_vol_20w",
    "vol_ratio",

    "return_1w",
    "return_2w",
    "return_4w",
    "return_8w",
    "return_12w",

    "drawdown_20w",
    "drawdown_52w",

    "obv_slope",

    "range_pct",
    "body_pct",
    "close_position"
]


print()
print(
    f"Features created: {len(FEATURES)}"
)


# ================================================================
# CLEAN FEATURES
# ================================================================

df[FEATURES] = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ================================================================
# MODEL DEFINITIONS
# ================================================================

MODELS = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=8,
            min_samples_split=10,
            subsample=0.8,
            loss="huber",
            random_state=42
        )
}


# ================================================================
# WALK-FORWARD EVALUATION
# ================================================================

print()
print("=" * 75)
print("WALK-FORWARD TARGET COMPARISON")
print("=" * 75)


results = []

all_predictions = []


# ================================================================
# LOOP THROUGH TARGETS
# ================================================================

for target_name, target_column in TARGETS.items():

    print()
    print()
    print("#" * 75)
    print(
        f"TARGET: {target_name}"
    )
    print("#" * 75)


    # ------------------------------------------------------------
    # Create target-specific dataset
    # ------------------------------------------------------------

    required = (
        ["Date", "Close"]
        + FEATURES
        + [target_column]
    )

    target_df = (
        df[required]
        .dropna()
        .reset_index(drop=True)
    )


    X = target_df[
        FEATURES
    ]

    y = target_df[
        target_column
    ]


    print(
        f"Usable observations: {len(target_df)}"
    )


    if len(target_df) < N_SPLITS + 1:

        print(
            "Not enough observations."
        )

        continue


    # ------------------------------------------------------------
    # TimeSeriesSplit
    # ------------------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )


    # ============================================================
    # LOOP THROUGH MODELS
    # ============================================================

    for model_name, model in MODELS.items():

        print()
        print(
            "-" * 70
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "-" * 70
        )


        fold_predictions = []

        fold_metrics = []


        # --------------------------------------------------------
        # FOLDS
        # --------------------------------------------------------

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            tscv.split(X),
            start=1
        ):

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]


            # ----------------------------------------------------
            # Fresh model
            # ----------------------------------------------------

            current_model = clone(
                model
            )


            # ----------------------------------------------------
            # Train
            # ----------------------------------------------------

            current_model.fit(
                X_train,
                y_train
            )


            # ----------------------------------------------------
            # Predict
            # ----------------------------------------------------

            predictions = (
                current_model
                .predict(X_test)
            )


            # ----------------------------------------------------
            # Metrics
            # ----------------------------------------------------

            mae = (
                mean_absolute_error(
                    y_test,
                    predictions
                )
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            r2 = (
                r2_score(
                    y_test,
                    predictions
                )
            )


            # ----------------------------------------------------
            # Store metrics
            # ----------------------------------------------------

            fold_metrics.append(
                {
                    "Fold":
                        fold,

                    "Train_Size":
                        len(train_idx),

                    "Test_Size":
                        len(test_idx),

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2
                }
            )


            # ----------------------------------------------------
            # Store predictions
            # ----------------------------------------------------

            prediction_part = pd.DataFrame(
                {
                    "Date":
                        target_df
                        .iloc[test_idx]
                        ["Date"]
                        .values,

                    "Target":
                        target_name,

                    "Model":
                        model_name,

                    "Fold":
                        fold,

                    "Actual":
                        y_test.values,

                    "Predicted":
                        predictions
                }
            )

            fold_predictions.append(
                prediction_part
            )


            print(
                f"Fold {fold}: "
                f"Train={len(train_idx)} | "
                f"Test={len(test_idx)} | "
                f"MAE={mae * 100:.3f}% | "
                f"RMSE={rmse * 100:.3f}% | "
                f"R²={r2:.4f}"
            )


        # ========================================================
        # COMBINE OOS PREDICTIONS
        # ========================================================

        prediction_df = pd.concat(
            fold_predictions,
            ignore_index=True
        )

        all_predictions.append(
            prediction_df
        )


        # ========================================================
        # OVERALL OOS METRICS
        # ========================================================

        actual = (
            prediction_df["Actual"]
            .to_numpy()
        )

        predicted = (
            prediction_df["Predicted"]
            .to_numpy()
        )


        overall_mae = (
            mean_absolute_error(
                actual,
                predicted
            )
        )

        overall_rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        overall_r2 = (
            r2_score(
                actual,
                predicted
            )
        )


        # --------------------------------------------------------
        # Correlation
        # --------------------------------------------------------

        if (
            np.std(actual) > 0
            and
            np.std(predicted) > 0
        ):

            correlation = (
                np.corrcoef(
                    actual,
                    predicted
                )[0, 1]
            )

        else:

            correlation = np.nan


        # --------------------------------------------------------
        # Direction diagnostic
        # --------------------------------------------------------

        actual_negative = (
            actual < 0
        )

        predicted_negative = (
            predicted < 0
        )

        direction_accuracy = (
            actual_negative
            ==
            predicted_negative
        ).mean()


        # ========================================================
        # FOLD SUMMARY
        # ========================================================

        fold_df = pd.DataFrame(
            fold_metrics
        )


        print()
        print(
            "OVERALL OOS:"
        )

        print(
            f"MAE        : "
            f"{overall_mae * 100:.3f}%"
        )

        print(
            f"RMSE       : "
            f"{overall_rmse * 100:.3f}%"
        )

        print(
            f"R²         : "
            f"{overall_r2:.4f}"
        )

        print(
            f"Correlation: "
            f"{correlation:.4f}"
        )

        print(
            f"Direction  : "
            f"{direction_accuracy * 100:.2f}%"
        )


        # ========================================================
        # STORE RESULT
        # ========================================================

        results.append(
            {
                "Target":
                    target_name,

                "Model":
                    model_name,

                "OOS_Observations":
                    len(actual),

                "MAE":
                    overall_mae,

                "RMSE":
                    overall_rmse,

                "R2":
                    overall_r2,

                "Correlation":
                    correlation,

                "Direction_Accuracy":
                    direction_accuracy,

                "Mean_Fold_MAE":
                    fold_df["MAE"]
                    .mean(),

                "Mean_Fold_RMSE":
                    fold_df["RMSE"]
                    .mean(),

                "Mean_Fold_R2":
                    fold_df["R2"]
                    .mean(),

                "Std_Fold_R2":
                    fold_df["R2"]
                    .std()
            }
        )


# ================================================================
# RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)


predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


df[
    ["Date", "Close"]
    + FEATURES
    + [
        "target_endpoint_return",
        "target_min_close_return",
        "target_min_low_return"
    ]
].to_csv(
    FEATURES_FILE,
    index=False
)


# ================================================================
# FINAL COMPARISON
# ================================================================

print()
print()
print("=" * 75)
print("FINAL TARGET + MODEL COMPARISON")
print("=" * 75)


display_results = (
    results_df.copy()
)

display_results["MAE_%"] = (
    display_results["MAE"]
    * 100
)

display_results["RMSE_%"] = (
    display_results["RMSE"]
    * 100
)

display_results["Direction_%"] = (
    display_results[
        "Direction_Accuracy"
    ]
    * 100
)


print(
    display_results[
        [
            "Target",
            "Model",
            "OOS_Observations",
            "MAE_%",
            "RMSE_%",
            "R2",
            "Correlation",
            "Direction_%"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ================================================================
# BEST MODEL FOR EACH TARGET
# ================================================================

print()
print("=" * 75)
print("BEST MODEL FOR EACH TARGET")
print("=" * 75)


for target_name in TARGETS.keys():

    subset = (
        results_df[
            results_df["Target"]
            == target_name
        ]
        .sort_values(
            "MAE"
        )
    )

    if len(subset) == 0:

        continue


    best = subset.iloc[0]


    print()
    print(
        target_name
    )

    print(
        f"Best model : "
        f"{best['Model']}"
    )

    print(
        f"MAE        : "
        f"{best['MAE'] * 100:.3f}%"
    )

    print(
        f"RMSE       : "
        f"{best['RMSE'] * 100:.3f}%"
    )

    print(
        f"R²         : "
        f"{best['R2']:.4f}"
    )

    print(
        f"Correlation: "
        f"{best['Correlation']:.4f}"
    )


# ================================================================
# BEST OVERALL CONFIGURATION
# ================================================================
#
# Primary selection criterion:
#   Lowest OOS MAE
#
# R² and correlation are displayed alongside it.
#
# ================================================================

best_overall = (
    results_df
    .sort_values(
        "MAE",
        ascending=True
    )
    .iloc[0]
)


print()
print("=" * 75)
print("BEST OVERALL CONFIGURATION BY OOS MAE")
print("=" * 75)

print(
    "Target:",
    best_overall["Target"]
)

print(
    "Model:",
    best_overall["Model"]
)

print(
    f"MAE : "
    f"{best_overall['MAE'] * 100:.3f}%"
)

print(
    f"RMSE: "
    f"{best_overall['RMSE'] * 100:.3f}%"
)

print(
    f"R²  : "
    f"{best_overall['R2']:.4f}"
)

print(
    f"Correlation: "
    f"{best_overall['Correlation']:.4f}"
)


# ================================================================
# IMPORTANT BASELINE
# ================================================================
#
# We compare the models against a very simple baseline:
#
# Predict the historical mean of the TRAINING SET.
#
# This is calculated separately inside each fold.
#
# If ML cannot beat this baseline, the features/models are not
# adding useful predictive information.
# ================================================================

print()
print("=" * 75)
print("BASELINE COMPARISON")
print("=" * 75)


baseline_results = []


for target_name, target_column in TARGETS.items():

    target_df = (
        df[
            ["Date"]
            + FEATURES
            + [target_column]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    X_target = target_df[
        FEATURES
    ]

    y_target = target_df[
        target_column
    ]


    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )


    baseline_actual = []

    baseline_predicted = []


    for train_idx, test_idx in tscv.split(
        X_target
    ):

        y_train = (
            y_target
            .iloc[train_idx]
        )

        y_test = (
            y_target
            .iloc[test_idx]
        )


        mean_prediction = (
            y_train.mean()
        )


        predictions = np.full(
            len(test_idx),
            mean_prediction
        )


        baseline_actual.extend(
            y_test.values
        )

        baseline_predicted.extend(
            predictions
        )


    baseline_actual = np.array(
        baseline_actual
    )

    baseline_predicted = np.array(
        baseline_predicted
    )


    baseline_mae = (
        mean_absolute_error(
            baseline_actual,
            baseline_predicted
        )
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            baseline_actual,
            baseline_predicted
        )
    )

    baseline_r2 = (
        r2_score(
            baseline_actual,
            baseline_predicted
        )
    )


    baseline_results.append(
        {
            "Target":
                target_name,

            "Baseline_MAE":
                baseline_mae,

            "Baseline_RMSE":
                baseline_rmse,

            "Baseline_R2":
                baseline_r2
        }
    )


baseline_df = pd.DataFrame(
    baseline_results
)


print(
    baseline_df[
        [
            "Target",
            "Baseline_MAE",
            "Baseline_RMSE",
            "Baseline_R2"
        ]
    ]
    .assign(
        Baseline_MAE=lambda x:
            x["Baseline_MAE"] * 100,

        Baseline_RMSE=lambda x:
            x["Baseline_RMSE"] * 100
    )
    .round(4)
    .to_string(
        index=False
    )
)


# ================================================================
# FINAL MESSAGE
# ================================================================

print()
print("=" * 75)
print("FILES CREATED")
print("=" * 75)

print(
    "1.",
    RESULTS_FILE
)

print(
    "2.",
    PREDICTIONS_FILE
)

print(
    "3.",
    FEATURES_FILE
)

print()
print("=" * 75)
print("V2 COMPLETED")
print("=" * 75)

print()
print(
    "IMPORTANT:"
)

print(
    "Use OOS MAE, RMSE, R² and correlation "
    "to judge the targets."
)

print(
    "Do NOT use Direction Accuracy alone."
)

print(
    "The best target is the one that provides "
    "useful out-of-sample predictive information, "
    "not simply the highest directional accuracy."
)

