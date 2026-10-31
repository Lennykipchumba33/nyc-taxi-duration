"""Train a leakage-controlled duration model on NYC TLC trip records."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Official TLC yellow-taxi Parquet file")
    parser.add_argument("--sample", type=int, default=50000)
    args = parser.parse_args()
    df = pd.read_parquet(args.input)
    pickup = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df["duration_seconds"] = (dropoff - pickup).dt.total_seconds()
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_weekday"] = pickup.dt.day_of_week
    features = ["pickup_hour", "pickup_weekday", "trip_distance", "passenger_count", "PULocationID"]
    df = df.dropna(subset=features + ["duration_seconds"]).copy()
    df = df.query("duration_seconds >= 60 and duration_seconds <= 7200 and trip_distance >= 0")
    df = df.sort_values("tpep_pickup_datetime")
    if len(df) > args.sample:
        df = df.iloc[:args.sample]
    split = max(1, int(len(df) * 0.8))
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_train, y_test = df["duration_seconds"].iloc[:split], df["duration_seconds"].iloc[split:]
    model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"rows={len(df)} test_rows={len(X_test)}")
    print(f"MAE seconds={mean_absolute_error(y_test, predictions):.1f}")
    print(f"RMSE seconds={root_mean_squared_error(y_test, predictions):.1f}")


if __name__ == "__main__":
    main()
