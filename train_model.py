# train_model.py
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess_and_features import make_daily_features

def train_model(pm_csv="pm25_data.csv", weather_csv="weather_data.csv"):
    # 1. Load saved CSVs
    print("📂 Loading data from CSVs...")
    pm_df = pd.read_csv(pm_csv)
    met_df = pd.read_csv(weather_csv)

    # 2. Preprocess + feature engineering
    print("⚙️ Preprocessing & feature engineering...")
    df = make_daily_features(pm_df, met_df)

    # 3. Train-test split
    X = df.drop(columns=["pm25", "date"])
    y = df["pm25"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 4. Train XGBoost model
    print("🚀 Training model...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"✅ Model trained! RMSE on test data: {rmse:.2f}")

    # 6. Save model + feature columns
    joblib.dump({"model": model, "features": X.columns.tolist()}, "model.joblib")
    print("💾 Saved model to model.joblib")

if __name__ == "__main__":
    train_model()
    