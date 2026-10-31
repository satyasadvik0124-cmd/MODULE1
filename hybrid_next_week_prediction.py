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

warnings.filterwarnings("ignore")

PROJECT_PATH = "/content/drive/MyDrive/market_ml_project-20260815T095833Z-1-001/market_ml_project"
INDEX_NAME = "BANKNIFTY"

GDELT_FILE = f"{PROJECT_PATH}/gdelt_weekly_sentiment.pkl"
RISK_FILE = f"{PROJECT_PATH}/banknifty_risk_engine_v2.csv"

FILES = {
    "NIFTY": "nifty_weekly_labeled.csv",
    "BANKNIFTY": "banknifty_weekly_labeled.csv",
    "SENSEX": "sensex_weekly_labeled.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{FILES[INDEX_NAME]}"

print("=" * 70)
print(f"HYBRID NEXT-WEEK PREDICTION — {INDEX_NAME}")
print("=" * 70)

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"Market rows: {len(df)}")
print(f"Market date range: {df.index.min().date()} → {df.index.max().date()}")

print("\nCreating technical features...")

df["price_range"] = (df["High"] - df["Low"]) / df["Close"]

df["close_position"] = (
    (df["Close"] - df["Low"]) /
    (df["High"] - df["Low"] + 1e-9)
)

df["gap"] = df["Open"] / df["Close"].shift(1) - 1

df["ema_20_slope"] = df["ema_20"].pct_change(2)
df["ema_50_slope"] = df["ema_50"].pct_change(4)

df["ema_20_50_cross"] = (
    df["ema_20"] > df["ema_50"]
).astype(int)

df["ema_50_100_cross"] = (
    df["ema_50"] > df["ema_100"]
).astype(int)

df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

df["rsi_mid"] = (
    (df["rsi_14"] >= 45) &
    (df["rsi_14"] <= 55)
).astype(int)

df["macd_cross"] = (
    df["macd"] > df["macd_signal"]
).astype(int)

df["macd_cross_change"] = df["macd_cross"].diff()

df["bb_squeeze"] = (
    df["bb_width"] <
    df["bb_width"].rolling(20).quantile(0.2)
).astype(int)

df["trend_strong"] = (df["adx"] > 25).astype(int)

df["di_cross"] = (
    df["plus_di"] > df["minus_di"]
).astype(int)

df["return_1w"] = df["Close"].pct_change(1)
df["return_2w"] = df["Close"].pct_change(2)
df["return_4w"] = df["Close"].pct_change(4)
df["return_8w"] = df["Close"].pct_change(8)
df["return_12w"] = df["Close"].pct_change(12)

df["mom_accel"] = df["return_1w"] - df["return_2w"]

df["label_lag1"] = df["label"].shift(1)
df["label_lag2"] = df["label"].shift(2)
df["label_lag3"] = df["label"].shift(3)

df["consec_up"] = df["label"].groupby(
    (df["label"] != df["label"].shift()).cumsum()
).cumcount()

df["consec_up"] = df["consec_up"] * df["label"]

df["obv_momentum"] = df["obv"].pct_change(4)

df["vol_20w"] = df["return_1w"].rolling(20).std()

df["vol_ratio"] = (
    df["return_1w"].rolling(4).std() /
    (df["vol_20w"] + 1e-9)
)

df["price_up_rsi_down"] = (
    (df["Close"] > df["Close"].shift(1)) &
    (df["rsi_14"] < df["rsi_14"].shift(1))
).astype(int)

print("\nLoading GDELT sentiment...")

gdelt = pd.read_pickle(GDELT_FILE)

if "week" in gdelt.columns:
    gdelt["week_end"] = pd.to_datetime(
        gdelt["week"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)
else:
    gdelt["week_end"] = pd.to_datetime(
        gdelt["Date"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

gdelt.dropna(subset=["week_end"], inplace=True)
gdelt.set_index("week_end", inplace=True)
gdelt.sort_index(inplace=True)

gdelt["sentiment_change"] = gdelt["mean_sentiment"].diff()
gdelt["sentiment_2w_change"] = gdelt["mean_sentiment"].diff(2)

gdelt["sentiment_4w_avg"] = (
    gdelt["mean_sentiment"].rolling(4).mean()
)

gdelt["sentiment_4w_std"] = (
    gdelt["mean_sentiment"].rolling(4).std()
)

gdelt["news_count_change"] = gdelt["news_count"].pct_change()

gdelt["news_count_4w_avg"] = (
    gdelt["news_count"].rolling(4).mean()
)

gdelt["sentiment_positive"] = (
    gdelt["mean_sentiment"] > 0
).astype(int)

gdelt["sentiment_negative"] = (
    gdelt["mean_sentiment"] < 0
).astype(int)

SENTIMENT_COLS = [
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

df.index = pd.to_datetime(df.index, errors="coerce")

try:
    df.index = df.index.tz_localize(None)
except:
    pass

df = df[~df.index.isna()]

market_reset = df.reset_index()
market_reset.rename(
    columns={market_reset.columns[0]: "Date"},
    inplace=True
)

sentiment_reset = gdelt[SENTIMENT_COLS].reset_index()
sentiment_reset.rename(
    columns={sentiment_reset.columns[0]: "Date"},
    inplace=True
)

sentiment_reset["Date"] = pd.to_datetime(
    sentiment_reset["Date"],
    errors="coerce"
)

sentiment_reset = (
    sentiment_reset
    .dropna(subset=["Date"])
    .sort_values("Date")
)

market_reset = market_reset.sort_values("Date")

df = pd.merge_asof(
    market_reset,
    sentiment_reset,
    on="Date",
    direction="backward"
)

df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"Rows after market + GDELT merge: {len(df)}")

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

X_all = df[feature_cols].copy()
y_all = df["label"].copy()

X_all = X_all.replace([np.inf, -np.inf], np.nan)
X_all = X_all.fillna(0)
X_all = X_all.astype("float64")
X_all = X_all.clip(lower=-1e6, upper=1e6)

print("\n" + "=" * 70)
print("TRAINING FINAL HYBRID MODEL")
print("=" * 70)

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

gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("gbc", gbc)
    ],
    voting="soft",
    weights=[0.5, 0.5]
)

rf.fit(X_all, y_all)
gbc.fit(X_all, y_all)
ensemble.fit(X_all, y_all)

print("RF trained.")
print("Gradient Boosting trained.")
print("Hybrid ensemble trained.")

latest_date = df.index.max()

latest_features = X_all.loc[[latest_date]]

prediction = ensemble.predict(latest_features)[0]
probabilities = ensemble.predict_proba(latest_features)[0]

prob_down = probabilities[0]
prob_up = probabilities[1]

if prediction == 1:
    direction = "UP"
    confidence = prob_up
else:
    direction = "DOWN"
    confidence = prob_down

print("\nLoading Risk Engine...")

risk_score = np.nan
risk_level = "Not Available"
risk_decision = "Not Available"
risk_reasons = "Not Available"

if os.path.exists(RISK_FILE):

    risk_df = pd.read_csv(
        RISK_FILE,
        parse_dates=["Date"]
    )

    risk_df.sort_values("Date", inplace=True)

    exact_risk = risk_df[
        risk_df["Date"] == pd.Timestamp(latest_date)
    ]

    if len(exact_risk) > 0:
        risk_row = exact_risk.iloc[-1]
    else:
        previous_risk = risk_df[
            risk_df["Date"] <= pd.Timestamp(latest_date)
        ]

        if len(previous_risk) > 0:
            risk_row = previous_risk.iloc[-1]
        else:
            risk_row = risk_df.iloc[-1]

    risk_score = float(risk_row["risk_score"])
    risk_level = str(risk_row["risk_level"])
    risk_decision = str(risk_row["risk_decision"])
    risk_reasons = str(risk_row["risk_reasons"])

else:
    print("WARNING: Risk Engine output file not found.")

INVESTOR_RISK_THRESHOLD = 50
CONFIDENCE_THRESHOLD = 0.60

if confidence < CONFIDENCE_THRESHOLD:
    final_signal = "LOW CONFIDENCE"
elif risk_score > INVESTOR_RISK_THRESHOLD:
    final_signal = "RISK EXCEEDS TOLERANCE"
elif direction == "UP":
    final_signal = "INVEST / STAY INVESTED"
else:
    final_signal = "REDUCE / EXIT"

print("\n" + "=" * 70)
print("NEXT-WEEK PREDICTION")
print("=" * 70)

print(f"Latest available week : {latest_date.date()}")
print("Prediction horizon    : NEXT WEEK")
print(f"Predicted direction   : {direction}")
print(f"UP probability        : {prob_up * 100:.2f}%")
print(f"DOWN probability      : {prob_down * 100:.2f}%")
print(f"Confidence            : {confidence * 100:.2f}%")

print("\n" + "=" * 70)
print("RISK ASSESSMENT")
print("=" * 70)

if pd.notna(risk_score):
    print(f"Risk score            : {risk_score:.0f}/100")
    print(f"Risk level            : {risk_level}")
    print(f"Investor threshold    : {INVESTOR_RISK_THRESHOLD}")
    print(f"Risk decision         : {risk_decision}")
    print(f"Risk reasons          : {risk_reasons}")
else:
    print("Risk assessment unavailable.")

print("\n" + "=" * 70)
print("FINAL INVESTOR SIGNAL")
print("=" * 70)
print(f"FINAL SIGNAL          : {final_signal}")
print("=" * 70)

result = pd.DataFrame([{
    "latest_week": latest_date,
    "prediction_horizon": "Next Week",
    "predicted_direction": direction,
    "probability_up": prob_up,
    "probability_down": prob_down,
    "confidence": confidence,
    "risk_score": risk_score,
    "risk_level": risk_level,
    "investor_risk_threshold": INVESTOR_RISK_THRESHOLD,
    "risk_decision": risk_decision,
    "risk_reasons": risk_reasons,
    "final_signal": final_signal
}])

OUTPUT_FILE = (
    f"{PROJECT_PATH}/"
    f"{INDEX_NAME.lower()}_next_week_prediction.csv"
)

result.to_csv(OUTPUT_FILE, index=False)

print("\nPrediction saved to:")
print(OUTPUT_FILE)

print("\nHybrid next-week prediction completed successfully.")
