
# ================================================================
# HYBRID WEEKLY MARKET DIRECTION PREDICTION
# TECHNICAL INDICATORS + GDELT SENTIMENT
#
# Leakage-free version
#
# Models:
#   1. Technical Random Forest
#   2. Technical Gradient Boosting
#   3. Technical Soft Voting Ensemble
#   4. GDELT Random Forest
#   5. GDELT Gradient Boosting
#   6. GDELT Soft Voting Ensemble
#
# Hybrid:
#   P_hybrid =
#       technical_weight * P_technical
#       + sentiment_weight * P_sentiment
#
# Tested weights:
#   100/0
#    90/10
#    80/20
#    70/30
#    60/40
#    50/50
#
# IMPORTANT:
#   The target label is NEVER used to construct the current-row
#   technical feature.
#
# ================================================================

import os
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


# ================================================================
# CONFIGURATION
# ================================================================

# ------------------------------------------------
# CHANGE ONLY THIS VALUE
# ------------------------------------------------
INDEX_NAME = "SENSEX"
# INDEX_NAME = "NIFTY"
# INDEX_NAME = "SENSEX"


# ------------------------------------------------
# PROJECT PATH
# ------------------------------------------------

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------
# DATA FILES
# ------------------------------------------------

MARKET_FILE = os.path.join(
    PROJECT_PATH,
    f"{INDEX_NAME.lower()}_weekly_labeled.csv"
)

GDELT_FILE = os.path.join(
    PROJECT_PATH,
    "gdelt_weekly_sentiment.pkl"
)


# ------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------

SPLIT_DATE = "2024-01-01"


# ------------------------------------------------
# RANDOM STATE
# ------------------------------------------------

RANDOM_STATE = 42


# ------------------------------------------------
# HYBRID WEIGHTS
# ------------------------------------------------

HYBRID_WEIGHTS = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50)
]


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# LOAD MARKET DATA
# ================================================================

def load_market_data():

    print_section("LOADING MARKET DATA")

    if not os.path.exists(MARKET_FILE):

        raise FileNotFoundError(
            f"Market file not found:\n{MARKET_FILE}"
        )

    df = pd.read_csv(MARKET_FILE)

    print("Market file:")
    print(MARKET_FILE)

    print("\nRows:", len(df))
    print("Columns:", len(df.columns))

    # ------------------------------------------------------------
    # DATE
    # ------------------------------------------------------------

    date_candidates = [
        "Date",
        "date",
        "Week",
        "week"
    ]

    date_col = None

    for col in date_candidates:

        if col in df.columns:

            date_col = col
            break

    if date_col is None:

        raise ValueError(
            "Could not find a date column in market CSV."
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[date_col]
    )

    df = df.sort_values(
        date_col
    ).reset_index(drop=True)

    if date_col != "Date":

        df.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    # ------------------------------------------------------------
    # LABEL
    # ------------------------------------------------------------

    if "label" not in df.columns:

        raise ValueError(
            "The market CSV does not contain the 'label' column."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = df["label"].astype(int)

    print(
        "\nMarket period:",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df["label"].value_counts().sort_index()
    )

    return df


# ================================================================
# LOAD GDELT DATA
# ================================================================

def load_gdelt_data():

    print_section("LOADING GDELT SENTIMENT DATA")

    if not os.path.exists(GDELT_FILE):

        raise FileNotFoundError(
            f"GDELT file not found:\n{GDELT_FILE}"
        )

    gdelt = pd.read_pickle(
        GDELT_FILE
    )

    print("GDELT file:")
    print(GDELT_FILE)

    print("\nRows:", len(gdelt))

    print(
        "Columns:",
        list(gdelt.columns)
    )

    # ------------------------------------------------------------
    # WEEK PARSING
    # ------------------------------------------------------------

    if "week" not in gdelt.columns:

        raise ValueError(
            "GDELT file does not contain 'week'."
        )

    def extract_week_end(value):

        value = str(value)

        # Format:
        # 2015-02-14/2015-02-20

        if "/" in value:

            return value.split("/")[-1]

        return value

    gdelt["week_end"] = (
        gdelt["week"]
        .astype(str)
        .apply(extract_week_end)
    )

    gdelt["Date"] = pd.to_datetime(
        gdelt["week_end"],
        errors="coerce"
    )

    gdelt = gdelt.dropna(
        subset=["Date"]
    )

    gdelt = gdelt.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        "\nGDELT period:",
        gdelt["Date"].min().date(),
        "to",
        gdelt["Date"].max().date()
    )

    return gdelt


# ================================================================
# CREATE NEXT-WEEK LABEL
# ================================================================

def create_label_if_needed(df):

    """
    This function is mainly a safety check.

    Your weekly_labels.py already creates:

        future_close = Close.shift(-1)
        label = future_close > Close

    Therefore the existing label is retained.

    We DO NOT recreate the label after merging because doing so
    could accidentally alter the original project target.
    """

    if "label" not in df.columns:

        raise ValueError(
            "No target label found."
        )

    return df


# ================================================================
# TECHNICAL FEATURE ENGINEERING
# ================================================================

def create_technical_features(df):

    print_section(
        "CREATING LEAKAGE-FREE TECHNICAL FEATURES"
    )

    df = df.copy()

    # ------------------------------------------------------------
    # Make sure chronological order is maintained
    # ------------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)


    # ============================================================
    # BASIC PRICE FEATURES
    # ============================================================

    if all(
        col in df.columns
        for col in ["High", "Low", "Close"]
    ):

        df["price_range"] = (
            df["High"] - df["Low"]
        )

        denominator = (
            df["High"] - df["Low"]
        ).replace(0, np.nan)

        df["close_position"] = (
            (df["Close"] - df["Low"])
            / denominator
        )


    # ============================================================
    # GAP
    # ============================================================

    if "Open" in df.columns:

        df["gap"] = (
            df["Open"]
            / df["Close"].shift(1)
            - 1
        )


    # ============================================================
    # EMA FEATURES
    # ============================================================

    if "ema_20" in df.columns:

        df["ema_20_slope"] = (
            df["ema_20"]
            .pct_change()
        )

    if "ema_50" in df.columns:

        df["ema_50_slope"] = (
            df["ema_50"]
            .pct_change()
        )

    if (
        "ema_20" in df.columns
        and "ema_50" in df.columns
    ):

        df["ema_20_50_cross"] = (
            df["ema_20"]
            - df["ema_50"]
        )

    if (
        "ema_50" in df.columns
        and "ema_100" in df.columns
    ):

        df["ema_50_100_cross"] = (
            df["ema_50"]
            - df["ema_100"]
        )


    # ============================================================
    # RSI
    # ============================================================

    if "rsi_14" in df.columns:

        df["rsi_overbought"] = (
            df["rsi_14"] > 70
        ).astype(int)

        df["rsi_oversold"] = (
            df["rsi_14"] < 30
        ).astype(int)

        df["rsi_mid"] = (
            (
                df["rsi_14"] >= 40
            )
            &
            (
                df["rsi_14"] <= 60
            )
        ).astype(int)

        df["rsi_change"] = (
            df["rsi_14"]
            .diff()
        )


    # ============================================================
    # MACD
    # ============================================================

    if (
        "macd" in df.columns
        and "macd_signal" in df.columns
    ):

        macd_difference = (
            df["macd"]
            - df["macd_signal"]
        )

        df["macd_cross"] = (
            macd_difference > 0
        ).astype(int)

        df["macd_cross_change"] = (
            df["macd_cross"]
            .diff()
        )


    # ============================================================
    # BOLLINGER BANDS
    # ============================================================

    if (
        "bb_width" in df.columns
    ):

        df["bb_squeeze"] = (
            df["bb_width"]
            < df["bb_width"]
            .rolling(20)
            .quantile(0.25)
        ).astype(int)


    # ============================================================
    # ADX / DI
    # ============================================================

    if (
        "plus_di" in df.columns
        and "minus_di" in df.columns
    ):

        df["di_cross"] = (
            df["plus_di"]
            > df["minus_di"]
        ).astype(int)

    if "adx" in df.columns:

        df["trend_strong"] = (
            df["adx"] > 25
        ).astype(int)


    # ============================================================
    # RETURNS
    # ============================================================

    if "Close" in df.columns:

        df["return_1w"] = (
            df["Close"]
            .pct_change(1)
        )

        df["return_2w"] = (
            df["Close"]
            .pct_change(2)
        )

        df["return_4w"] = (
            df["Close"]
            .pct_change(4)
        )

        df["return_8w"] = (
            df["Close"]
            .pct_change(8)
        )

        df["return_12w"] = (
            df["Close"]
            .pct_change(12)
        )

        df["mom_accel"] = (
            df["return_1w"]
            - df["return_4w"]
            / 4
        )


    # ============================================================
    # PRIOR LABEL FEATURES
    #
    # IMPORTANT:
    #
    # label_lag1 = previous week's target
    # label_lag2 = target from two weeks ago
    # label_lag3 = target from three weeks ago
    #
    # These are valid because they use ONLY past labels.
    # ============================================================

    df["label_lag1"] = (
        df["label"].shift(1)
    )

    df["label_lag2"] = (
        df["label"].shift(2)
    )

    df["label_lag3"] = (
        df["label"].shift(3)
    )


    # ============================================================
    # LEAKAGE-FREE CONSECUTIVE UP FEATURE
    #
    # OLD / WRONG:
    #
    # consec_up based directly on current label
    #
    # That leaks label[t] into feature[t].
    #
    # CORRECT:
    #
    # Use label.shift(1), i.e. previous weeks only.
    # ============================================================

    previous_label = (
        df["label"].shift(1)
    )

    groups = (
        previous_label
        .ne(previous_label.shift())
        .cumsum()
    )

    df["consec_up"] = (
        previous_label
        .groupby(groups)
        .cumcount()
    )

    df["consec_up"] = (
        df["consec_up"]
        * previous_label.fillna(0)
    )


    # ============================================================
    # OBV MOMENTUM
    # ============================================================

    if "obv" in df.columns:

        df["obv_momentum"] = (
            df["obv"]
            .pct_change()
        )

    elif "Volume" in df.columns:

        direction = np.sign(
            df["Close"].diff()
        )

        obv = (
            direction
            * df["Volume"]
        ).fillna(0).cumsum()

        df["obv_momentum"] = (
            obv.pct_change()
        )


    # ============================================================
    # VOLATILITY
    # ============================================================

    if "Close" in df.columns:

        weekly_returns = (
            df["Close"]
            .pct_change()
        )

        df["vol_20w"] = (
            weekly_returns
            .rolling(20)
            .std()
        )

    if (
        "Volume" in df.columns
    ):

        volume_average = (
            df["Volume"]
            .rolling(20)
            .mean()
        )

        df["vol_ratio"] = (
            df["Volume"]
            / volume_average
        )


    # ============================================================
    # PRICE / RSI DIVERGENCE
    # ============================================================

    if (
        "Close" in df.columns
        and "rsi_14" in df.columns
    ):

        price_change = (
            df["Close"]
            .pct_change()
        )

        rsi_change = (
            df["rsi_14"]
            .diff()
        )

        df["price_up_rsi_down"] = (
            (price_change > 0)
            &
            (rsi_change < 0)
        ).astype(int)


    # ============================================================
    # DROP INF / NAN
    # ============================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ================================================================
# GET TECHNICAL FEATURES
# ================================================================

def get_technical_feature_columns(df):

    exclude_columns = {
        "Date",
        "label",
        "future_close",
        "week",
        "week_end"
    }

    feature_columns = []

    for column in df.columns:

        if column in exclude_columns:
            continue

        # Only numeric columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            feature_columns.append(
                column
            )

    return feature_columns


# ================================================================
# CREATE GDELT FEATURES
# ================================================================

def create_gdelt_features(gdelt):

    print_section(
        "CREATING GDELT FEATURES"
    )

    gdelt = gdelt.copy()

    gdelt = gdelt.sort_values(
        "Date"
    ).reset_index(drop=True)


    required = [
        "mean_sentiment",
        "median_sentiment",
        "sentiment_std",
        "news_count"
    ]

    missing = [
        col
        for col in required
        if col not in gdelt.columns
    ]

    if missing:

        raise ValueError(
            "Missing GDELT columns: "
            + str(missing)
        )


    # ============================================================
    # SENTIMENT CHANGE
    # ============================================================

    gdelt["sentiment_change"] = (
        gdelt["mean_sentiment"]
        .diff()
    )

    gdelt["sentiment_2w_change"] = (
        gdelt["mean_sentiment"]
        .diff(2)
    )


    # ============================================================
    # ROLLING SENTIMENT
    # ============================================================

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


    # ============================================================
    # NEWS COUNT
    # ============================================================

    gdelt["news_count_change"] = (
        gdelt["news_count"]
        .pct_change()
    )

    gdelt["news_count_4w_avg"] = (
        gdelt["news_count"]
        .rolling(4)
        .mean()
    )


    gdelt = gdelt.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return gdelt


# ================================================================
# MERGE MARKET + GDELT
# ================================================================

def merge_datasets(market, gdelt):

    print_section(
        "MERGING MARKET + GDELT"
    )

    # ------------------------------------------------------------
    # Technical features first
    # ------------------------------------------------------------

    market = create_technical_features(
        market
    )

    # ------------------------------------------------------------
    # GDELT features
    # ------------------------------------------------------------

    gdelt = create_gdelt_features(
        gdelt
    )


    # ------------------------------------------------------------
    # Keep only the required GDELT columns
    # ------------------------------------------------------------

    gdelt_columns = [
        "Date",
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

    gdelt = gdelt[
        gdelt_columns
    ].copy()


    # ------------------------------------------------------------
    # INNER JOIN
    #
    # This is important.
    #
    # Both models must make predictions on exactly the same
    # weeks so their probabilities can be fused fairly.
    # ------------------------------------------------------------

    merged = pd.merge(
        market,
        gdelt,
        on="Date",
        how="inner"
    )


    merged = merged.sort_values(
        "Date"
    ).reset_index(drop=True)


    # ------------------------------------------------------------
    # Remove rows with unavailable engineered features
    # ------------------------------------------------------------

    merged = merged.replace(
        [np.inf, -np.inf],
        np.nan
    )

    merged = merged.dropna(
        subset=["label"]
    )


    print(
        "\nMarket rows:",
        len(market)
    )

    print(
        "GDELT rows:",
        len(gdelt)
    )

    print(
        "Common rows:",
        len(merged)
    )

    print(
        "Common period:",
        merged["Date"].min().date(),
        "to",
        merged["Date"].max().date()
    )


    return merged


# ================================================================
# PREPARE FEATURES
# ================================================================

def prepare_features(df):

    technical_features = (
        get_technical_feature_columns(
            df
        )
    )


    # ------------------------------------------------------------
    # Explicitly remove GDELT columns from technical model
    # ------------------------------------------------------------

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

    technical_features = [
        col
        for col in technical_features
        if col not in sentiment_features
    ]


    # ------------------------------------------------------------
    # Remove target / non-feature columns
    # ------------------------------------------------------------

    forbidden = [
        "label",
        "future_close"
    ]

    technical_features = [
        col
        for col in technical_features
        if col not in forbidden
    ]


    # ------------------------------------------------------------
    # GDELT feature list
    # ------------------------------------------------------------

    sentiment_features = [
        col
        for col in sentiment_features
        if col in df.columns
    ]


    print_section(
        "FEATURE SUMMARY"
    )

    print(
        "Technical features:",
        len(technical_features)
    )

    print(
        "GDELT features:",
        len(sentiment_features)
    )

    print(
        "\nTechnical:"
    )

    print(
        technical_features
    )

    print(
        "\nGDELT:"
    )

    print(
        sentiment_features
    )


    return (
        technical_features,
        sentiment_features
    )


# ================================================================
# IMPUTE FEATURES
# ================================================================

def clean_features(
    X_train,
    X_test
):

    # Fit medians only on training data.
    # This prevents test-set information from entering
    # preprocessing.

    medians = (
        X_train
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .median()
    )

    X_train = (
        X_train
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(medians)
    )

    X_test = (
        X_test
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(medians)
    )

    return X_train, X_test


# ================================================================
# CREATE MODELS
# ================================================================

def create_rf():

    return RandomForestClassifier(

        n_estimators=1000,

        max_depth=8,

        min_samples_leaf=5,

        min_samples_split=10,

        max_features="sqrt",

        class_weight="balanced",

        oob_score=True,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


def create_gbc():

    return GradientBoostingClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.80,

        min_samples_leaf=10,

        random_state=RANDOM_STATE
    )


def create_ensemble():

    rf = create_rf()

    gbc = create_gbc()

    ensemble = VotingClassifier(

        estimators=[
            ("rf", rf),
            ("gbc", gbc)
        ],

        voting="soft",

        weights=[
            0.5,
            0.5
        ],

        n_jobs=-1
    )

    return ensemble


# ================================================================
# TRAIN MODEL SET
# ================================================================

def train_model_set(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name
):

    print_section(
        f"TRAINING {model_name.upper()} MODELS"
    )


    # ------------------------------------------------------------
    # RF
    # ------------------------------------------------------------

    rf = create_rf()

    rf.fit(
        X_train,
        y_train
    )

    rf_probability = rf.predict_proba(
        X_test
    )[:, 1]

    rf_prediction = (
        rf_probability >= 0.5
    ).astype(int)

    rf_accuracy = accuracy_score(
        y_test,
        rf_prediction
    )


    # ------------------------------------------------------------
    # GBC
    # ------------------------------------------------------------

    gbc = create_gbc()

    gbc.fit(
        X_train,
        y_train
    )

    gbc_probability = gbc.predict_proba(
        X_test
    )[:, 1]

    gbc_prediction = (
        gbc_probability >= 0.5
    ).astype(int)

    gbc_accuracy = accuracy_score(
        y_test,
        gbc_prediction
    )


    # ------------------------------------------------------------
    # SOFT VOTING ENSEMBLE
    # ------------------------------------------------------------

    ensemble = create_ensemble()

    ensemble.fit(
        X_train,
        y_train
    )

    ensemble_probability = (
        ensemble.predict_proba(
            X_test
        )[:, 1]
    )

    ensemble_prediction = (
        ensemble_probability >= 0.5
    ).astype(int)

    ensemble_accuracy = accuracy_score(
        y_test,
        ensemble_prediction
    )


    print(
        f"\n{model_name} RF accuracy:"
        f" {rf_accuracy * 100:.2f}%"
    )

    print(
        f"{model_name} GBC accuracy:"
        f" {gbc_accuracy * 100:.2f}%"
    )

    print(
        f"{model_name} Ensemble accuracy:"
        f" {ensemble_accuracy * 100:.2f}%"
    )


    if hasattr(rf, "oob_score_"):

        print(
            f"{model_name} RF OOB:"
            f" {rf.oob_score_ * 100:.2f}%"
        )

    return {

        "rf": rf,

        "gbc": gbc,

        "ensemble": ensemble,

        "rf_probability": rf_probability,

        "gbc_probability": gbc_probability,

        "ensemble_probability":
            ensemble_probability,

        "rf_accuracy":
            rf_accuracy,

        "gbc_accuracy":
            gbc_accuracy,

        "ensemble_accuracy":
            ensemble_accuracy
    }


# ================================================================
# HYBRID PROBABILITY FUSION
# ================================================================

def calculate_hybrid_results(
    y_test,
    technical_probability,
    sentiment_probability
):

    print_section(
        "HYBRID PROBABILITY FUSION"
    )


    results = []

    for (
        technical_weight,
        sentiment_weight
    ) in HYBRID_WEIGHTS:

        # --------------------------------------------------------
        # Probability-level fusion
        # --------------------------------------------------------

        hybrid_probability = (

            technical_weight
            * technical_probability

            +

            sentiment_weight
            * sentiment_probability
        )


        # --------------------------------------------------------
        # Convert probability to direction
        # --------------------------------------------------------

        hybrid_prediction = (
            hybrid_probability >= 0.5
        ).astype(int)


        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            hybrid_prediction
        )


        cm = confusion_matrix(
            y_test,
            hybrid_prediction,
            labels=[0, 1]
        )


        report = classification_report(
            y_test,
            hybrid_prediction,
            output_dict=True,
            zero_division=0
        )


        precision = report["1"]["precision"]

        recall = report["1"]["recall"]

        f1 = report["1"]["f1-score"]


        results.append({

            "technical_weight":
                technical_weight,

            "sentiment_weight":
                sentiment_weight,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "tn":
                cm[0, 0],

            "fp":
                cm[0, 1],

            "fn":
                cm[1, 0],

            "tp":
                cm[1, 1]
        })


        print(
            f"\nTechnical {technical_weight:.0%}"
            f" / Sentiment {sentiment_weight:.0%}"
        )

        print(
            f"Accuracy : {accuracy * 100:.2f}%"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            "Confusion matrix:"
        )

        print(
            cm
        )


    return pd.DataFrame(
        results
    )


# ================================================================
# WALK-FORWARD HYBRID VALIDATION
# ================================================================

def walk_forward_hybrid(
    X_technical,
    X_sentiment,
    y,
    n_splits=5
):

    print_section(
        "5-FOLD WALK-FORWARD HYBRID VALIDATION"
    )


    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )


    all_results = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        tscv.split(X_technical),
        start=1
    ):

        print(
            f"\n----- FOLD {fold} -----"
        )


        Xtech_train = (
            X_technical.iloc[
                train_idx
            ]
        )

        Xtech_test = (
            X_technical.iloc[
                test_idx
            ]
        )


        Xsent_train = (
            X_sentiment.iloc[
                train_idx
            ]
        )

        Xsent_test = (
            X_sentiment.iloc[
                test_idx
            ]
        )


        y_train = (
            y.iloc[
                train_idx
            ]
        )

        y_test = (
            y.iloc[
                test_idx
            ]
        )


        # --------------------------------------------------------
        # CLEAN USING TRAINING MEDIANS
        # --------------------------------------------------------

        Xtech_train, Xtech_test = clean_features(
            Xtech_train,
            Xtech_test
        )

        Xsent_train, Xsent_test = clean_features(
            Xsent_train,
            Xsent_test
        )


        # --------------------------------------------------------
        # TECHNICAL ENSEMBLE
        # --------------------------------------------------------

        technical_model = (
            create_ensemble()
        )

        technical_model.fit(
            Xtech_train,
            y_train
        )

        technical_probability = (
            technical_model
            .predict_proba(
                Xtech_test
            )[:, 1]
        )


        # --------------------------------------------------------
        # SENTIMENT ENSEMBLE
        # --------------------------------------------------------

        sentiment_model = (
            create_ensemble()
        )

        sentiment_model.fit(
            Xsent_train,
            y_train
        )

        sentiment_probability = (
            sentiment_model
            .predict_proba(
                Xsent_test
            )[:, 1]
        )


        # --------------------------------------------------------
        # EACH HYBRID WEIGHT
        # --------------------------------------------------------

        for (
            technical_weight,
            sentiment_weight
        ) in HYBRID_WEIGHTS:

            hybrid_probability = (

                technical_weight
                * technical_probability

                +

                sentiment_weight
                * sentiment_probability
            )


            prediction = (
                hybrid_probability >= 0.5
            ).astype(int)


            accuracy = accuracy_score(
                y_test,
                prediction
            )


            all_results.append({

                "fold":
                    fold,

                "technical_weight":
                    technical_weight,

                "sentiment_weight":
                    sentiment_weight,

                "accuracy":
                    accuracy
            })


            print(
                f"{technical_weight:.0%}/"
                f"{sentiment_weight:.0%}"
                f" -> "
                f"{accuracy * 100:.2f}%"
            )


    results_df = pd.DataFrame(
        all_results
    )


    summary = (
        results_df
        .groupby(
            [
                "technical_weight",
                "sentiment_weight"
            ]
        )["accuracy"]
        .agg(
            [
                "mean",
                "std"
            ]
        )
        .reset_index()
    )


    print_section(
        "WALK-FORWARD SUMMARY"
    )


    for _, row in summary.iterrows():

        print(
            f"Technical "
            f"{row['technical_weight']:.0%}"
            f" / Sentiment "
            f"{row['sentiment_weight']:.0%}"
            f" -> "
            f"{row['mean'] * 100:.2f}% "
            f"+/- "
            f"{row['std'] * 100:.2f}%"
        )


    return (
        results_df,
        summary
    )


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

def show_feature_importance(
    model,
    feature_names,
    model_name
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return


    importance = pd.DataFrame({

        "feature":
            feature_names,

        "importance":
            model.feature_importances_
    })


    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )


    print_section(
        f"{model_name.upper()} FEATURE IMPORTANCE"
    )


    print(
        importance.head(20)
        .to_string(index=False)
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print_section(
        f"HYBRID MARKET PREDICTION - {INDEX_NAME}"
    )


    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    market = load_market_data()

    gdelt = load_gdelt_data()


    # ============================================================
    # 2. MERGE
    # ============================================================

    data = merge_datasets(
        market,
        gdelt
    )


    # ============================================================
    # 3. FEATURES
    # ============================================================

    (
        technical_features,
        sentiment_features
    ) = prepare_features(
        data
    )


    # ============================================================
    # 4. REMOVE ROWS WHERE FEATURES ARE UNAVAILABLE
    #
    # We need both technical and sentiment information.
    # ============================================================

    required_columns = (
        technical_features
        +
        sentiment_features
        +
        ["label"]
    )


    data = data.dropna(
        subset=required_columns
    ).reset_index(
        drop=True
    )


    print_section(
        "FINAL COMMON DATASET"
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "Period:",
        data["Date"].min().date(),
        "to",
        data["Date"].max().date()
    )


    # ============================================================
    # 5. CHRONOLOGICAL TRAIN / TEST SPLIT
    # ============================================================

    train_mask = (
        data["Date"]
        < pd.Timestamp(
            SPLIT_DATE
        )
    )

    test_mask = (
        data["Date"]
        >= pd.Timestamp(
            SPLIT_DATE
        )
    )


    train_data = (
        data.loc[
            train_mask
        ]
        .copy()
    )

    test_data = (
        data.loc[
            test_mask
        ]
        .copy()
    )


    print_section(
        "TRAIN / TEST SPLIT"
    )

    print(
        "Training rows:",
        len(train_data)
    )

    print(
        "Testing rows:",
        len(test_data)
    )


    print(
        "\nTraining period:"
    )

    print(
        train_data["Date"].min().date(),
        "to",
        train_data["Date"].max().date()
    )


    print(
        "\nTesting period:"
    )

    print(
        test_data["Date"].min().date(),
        "to",
        test_data["Date"].max().date()
    )


    # ============================================================
    # 6. TARGET
    # ============================================================

    y_train = (
        train_data["label"]
        .astype(int)
    )

    y_test = (
        test_data["label"]
        .astype(int)
    )


    print(
        "\nTraining labels:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )


    print(
        "\nTesting labels:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )


    # ============================================================
    # 7. FEATURES
    # ============================================================

    Xtech_train = (
        train_data[
            technical_features
        ]
        .copy()
    )

    Xtech_test = (
        test_data[
            technical_features
        ]
        .copy()
    )


    Xsent_train = (
        train_data[
            sentiment_features
        ]
        .copy()
    )

    Xsent_test = (
        test_data[
            sentiment_features
        ]
        .copy()
    )


    # ============================================================
    # 8. CLEAN FEATURES
    # ============================================================

    (
        Xtech_train,
        Xtech_test
    ) = clean_features(
        Xtech_train,
        Xtech_test
    )


    (
        Xsent_train,
        Xsent_test
    ) = clean_features(
        Xsent_train,
        Xsent_test
    )


    # ============================================================
    # 9. TRAIN TECHNICAL MODELS
    # ============================================================

    technical_models = train_model_set(

        Xtech_train,

        y_train,

        Xtech_test,

        y_test,

        "Technical"
    )


    # ============================================================
    # 10. TRAIN GDELT MODELS
    # ============================================================

    sentiment_models = train_model_set(

        Xsent_train,

        y_train,

        Xsent_test,

        y_test,

        "GDELT Sentiment"
    )


    # ============================================================
    # 11. FEATURE IMPORTANCE
    # ============================================================

    show_feature_importance(

        technical_models["rf"],

        technical_features,

        "Technical RF"
    )


    show_feature_importance(

        sentiment_models["rf"],

        sentiment_features,

        "GDELT RF"
    )


    # ============================================================
    # 12. HYBRID
    #
    # IMPORTANT:
    #
    # We use the SOFT VOTING ENSEMBLE probabilities from BOTH
    # branches.
    #
    # This means:
    #
    # Technical probability =
    #     RF + GBC ensemble
    #
    # Sentiment probability =
    #     RF + GBC ensemble
    #
    # Then:
    #
    # P_hybrid =
    #     WT * P_technical
    #     +
    #     WS * P_sentiment
    # ============================================================

    hybrid_results = calculate_hybrid_results(

        y_test,

        technical_models[
            "ensemble_probability"
        ],

        sentiment_models[
            "ensemble_probability"
        ]
    )


    # ============================================================
    # 13. FIND BEST HYBRID CONFIGURATION
    #
    # NOTE:
    # This is NOT "best" in a scientific/general sense.
    # It is simply the highest accuracy among the tested
    # configurations on this fixed test period.
    # ============================================================

    best_row = (
        hybrid_results
        .sort_values(
            "accuracy",
            ascending=False
        )
        .iloc[0]
    )


    print_section(
        "FIXED TEST PERIOD RESULT"
    )


    print(
        f"Highest test accuracy among tested "
        f"weights: "
        f"{best_row['accuracy'] * 100:.2f}%"
    )


    print(
        f"Technical weight: "
        f"{best_row['technical_weight']:.0%}"
    )


    print(
        f"Sentiment weight: "
        f"{best_row['sentiment_weight']:.0%}"
    )


    # ============================================================
    # 14. WALK-FORWARD
    # ============================================================

    (
        walk_forward_results,
        walk_forward_summary
    ) = walk_forward_hybrid(

        data[
            technical_features
        ].reset_index(drop=True),

        data[
            sentiment_features
        ].reset_index(drop=True),

        data[
            "label"
        ].reset_index(drop=True),

        n_splits=5
    )


    # ============================================================
    # 15. SAVE RESULTS
    # ============================================================

    output_file = os.path.join(

        PROJECT_PATH,

        f"{INDEX_NAME.lower()}_hybrid_results.csv"
    )


    hybrid_results.to_csv(
        output_file,
        index=False
    )


    walk_forward_file = os.path.join(

        PROJECT_PATH,

        f"{INDEX_NAME.lower()}_hybrid_walkforward.csv"
    )


    walk_forward_results.to_csv(
        walk_forward_file,
        index=False
    )


    walk_forward_summary_file = os.path.join(

        PROJECT_PATH,

        f"{INDEX_NAME.lower()}_hybrid_walkforward_summary.csv"
    )


    walk_forward_summary.to_csv(
        walk_forward_summary_file,
        index=False
    )


    # ============================================================
    # 16. SAVE TEST PREDICTIONS
    # ============================================================

    prediction_output = test_data[
        [
            "Date",
            "label"
        ]
    ].copy()


    prediction_output[
        "technical_probability"
    ] = technical_models[
        "ensemble_probability"
    ]


    prediction_output[
        "sentiment_probability"
    ] = sentiment_models[
        "ensemble_probability"
    ]


    # Add each hybrid probability

    for (
        technical_weight,
        sentiment_weight
    ) in HYBRID_WEIGHTS:

        weight_name = (
            f"{int(technical_weight * 100)}_"
            f"{int(sentiment_weight * 100)}"
        )


        probability = (

            technical_weight
            * prediction_output[
                "technical_probability"
            ]

            +

            sentiment_weight
            * prediction_output[
                "sentiment_probability"
            ]
        )


        prediction_output[
            f"hybrid_probability_{weight_name}"
        ] = probability


        prediction_output[
            f"hybrid_prediction_{weight_name}"
        ] = (
            probability >= 0.5
        ).astype(int)


    prediction_file = os.path.join(

        PROJECT_PATH,

        f"{INDEX_NAME.lower()}_hybrid_predictions.csv"
    )


    prediction_output.to_csv(
        prediction_file,
        index=False
    )


    # ============================================================
    # 17. FINAL SUMMARY
    # ============================================================

    print_section(
        "FINAL SUMMARY"
    )


    print(
        f"\nINDEX: {INDEX_NAME}"
    )


    print(
        "\nTechnical RF:"
        f" {technical_models['rf_accuracy'] * 100:.2f}%"
    )


    print(
        "Technical GBC:"
        f" {technical_models['gbc_accuracy'] * 100:.2f}%"
    )


    print(
        "Technical Ensemble:"
        f" {technical_models['ensemble_accuracy'] * 100:.2f}%"
    )


    print(
        "\nGDELT RF:"
        f" {sentiment_models['rf_accuracy'] * 100:.2f}%"
    )


    print(
        "GDELT GBC:"
        f" {sentiment_models['gbc_accuracy'] * 100:.2f}%"
    )


    print(
        "GDELT Ensemble:"
        f" {sentiment_models['ensemble_accuracy'] * 100:.2f}%"
    )


    print(
        "\nHybrid test configurations:"
    )

    print(
        hybrid_results[
            [
                "technical_weight",
                "sentiment_weight",
                "accuracy"
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nWalk-forward:"
    )

    print(
        walk_forward_summary.to_string(
            index=False
        )
    )


    print(
        "\nFiles saved:"
    )

    print(
        output_file
    )

    print(
        walk_forward_file
    )

    print(
        walk_forward_summary_file
    )

    print(
        prediction_file
    )


    print_section(
        "COMPLETED"
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()

