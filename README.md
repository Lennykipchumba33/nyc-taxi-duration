# NYC Taxi Trip Duration

[![CI](https://github.com/Lennykipchumba33/nyc-taxi-duration/actions/workflows/ci.yml/badge.svg)](https://github.com/Lennykipchumba33/nyc-taxi-duration/actions/workflows/ci.yml)

A leakage-controlled regression example using official NYC Taxi & Limousine Commission (TLC) trip records. The model predicts completed trip duration from features intended to be available at pickup time.

## What this demonstrates

- Reproducible ingestion of Parquet data with pandas.
- Time-derived feature engineering from pickup timestamps.
- Chronological train/test evaluation rather than a random split.
- Leakage controls: drop-off and post-trip fare fields are not used as predictors.
- Regression evaluation with MAE and RMSE.

## Pipeline

```text
TLC Parquet -> validation/filtering -> pickup-time features -> chronological split
            -> HistGradientBoostingRegressor -> MAE/RMSE
```

Features used in the compact example are pickup hour, pickup weekday, trip distance, passenger count, and pickup taxi zone. Taxi-zone IDs are treated as numeric codes for demonstration; a production system should use a categorical or geographic representation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
mkdir -p data
curl -L -o data/yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
python -m src.train --input data/yellow_tripdata_2024-01.parquet --sample 50000
```

The command prints the number of rows, test rows, MAE in seconds, and RMSE in seconds. CI validates the Python source without downloading the large dataset.

## Project layout

- `src/train.py` — data loading, feature engineering, training, and evaluation.
- `.github/workflows/ci.yml` — compile and dependency checks.
- `requirements.txt` — runtime dependencies.

## Data and limitations

Use the official [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). TLC notes that the records are supplied by authorized technology providers and that accuracy and completeness are not guaranteed. The [NYC OpenData Terms of Use](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje/) apply to public data access.

These observational administrative records are not a random sample of all NYC travel and do not support causal claims. Schemas can change, and production use would require monitoring, geographic encoding, validation of outliers, and a documented model review process.

## License and attribution

The code is MIT-licensed. Dataset citation: New York City Taxi & Limousine Commission, “TLC Trip Record Data” and “TLC Trip Records User Guide.” Dataset access remains subject to the official source terms.
