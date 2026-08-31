import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("\n========================================")
print("   MARKETMIND AI - REVENUE PREDICTION")
print("========================================")


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/marketing_kpi_data.csv")

df["Date"] = pd.to_datetime(df["Date"])


# ==========================================
# 2. CREATE DAILY DATA
# ==========================================

daily = (
    df.groupby("Date")
    .agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum")
    )
    .reset_index()
)


# ==========================================
# 3. CREATE FEATURES
# ==========================================

daily["Day"] = daily["Date"].dt.day
daily["Month"] = daily["Date"].dt.month
daily["DayOfWeek"] = daily["Date"].dt.dayofweek

daily["CTR"] = (
    daily["Clicks"] /
    daily["Impressions"]
) * 100

daily["Conversion_Rate"] = (
    daily["Conversions"] /
    daily["Clicks"]
) * 100

daily["ROAS"] = (
    daily["Revenue"] /
    daily["Spend"]
)


# ==========================================
# 4. REMOVE INVALID VALUES
# ==========================================

daily = daily.replace(
    [np.inf, -np.inf],
    np.nan
)

daily = daily.dropna()


# ==========================================
# 5. FEATURES AND TARGET
# ==========================================

features = [
    "Impressions",
    "Clicks",
    "Conversions",
    "Spend",
    "CTR",
    "Conversion_Rate",
    "ROAS",
    "Day",
    "Month",
    "DayOfWeek"
]

X = daily[features]

y = daily["Revenue"]


# ==========================================
# 6. TIME-BASED TRAIN TEST SPLIT
# ==========================================

split_index = int(len(daily) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining records :", len(X_train))
print("Testing records  :", len(X_test))


# ==========================================
# 7. CREATE ML MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


print("\nModel training completed!")


# ==========================================
# 9. MAKE TEST PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========================================")
print("           MODEL PERFORMANCE")
print("========================================")

print(f"\nMAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 11. PREDICTION TABLE
# ==========================================

prediction_data = pd.DataFrame({
    "Date": daily.iloc[split_index:]["Date"].values,
    "Actual_Revenue": y_test.values,
    "Predicted_Revenue": predictions
})


prediction_data["Difference"] = (
    prediction_data["Actual_Revenue"]
    -
    prediction_data["Predicted_Revenue"]
)


prediction_data = prediction_data.round(2)


# ==========================================
# 12. DISPLAY PREDICTIONS
# ==========================================

print("\n========================================")
print("        SAMPLE PREDICTIONS")
print("========================================\n")

print(
    prediction_data.head(10).to_string(
        index=False
    )
)


# ==========================================
# 13. NEXT REVENUE PREDICTION
# ==========================================

latest_data = X.iloc[[-1]]

next_prediction = model.predict(
    latest_data
)[0]


print("\n========================================")
print("          WHAT'S NEXT?")
print("========================================")

print(
    f"\nPredicted Next Revenue: ₹{next_prediction:,.2f}"
)


# ==========================================
# 14. SAVE PREDICTIONS
# ==========================================

prediction_data.to_csv(
    "data/revenue_predictions.csv",
    index=False
)


# ==========================================
# 15. SAVE MODEL
# ==========================================

import joblib

joblib.dump(
    model,
    "models/revenue_model.pkl"
)


print("\n========================================")
print("Prediction files saved successfully!")
print("========================================")

print("\ndata/revenue_predictions.csv")
print("models/revenue_model.pkl")