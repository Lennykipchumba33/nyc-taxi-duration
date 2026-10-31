# NYC Taxi Trip Duration

A real-world regression project using official NYC Taxi & Limousine Commission (TLC) trip records. It predicts completed trip duration from information intended to be available at pickup time.

## Data source

Download a monthly yellow-taxi Parquet file from the [official TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Example: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet`.

TLC states that each row represents a trip by a TLC-licensed vehicle. The records are supplied by authorized technology providers, and TLC warns that accuracy and completeness are not guaranteed. The [NYC OpenData Terms of Use](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje/) apply to public data access. Review those terms before redistribution or production use.

Citation: New York City Taxi & Limousine Commission, “TLC Trip Record Data,” and “TLC Trip Records User Guide.”

## Run

```bash
pip install -r requirements.txt
python -m src.train --input data/yellow_tripdata_2024-01.parquet --sample 50000
```

The script uses chronological evaluation, reports MAE/RMSE, and excludes post-trip fare and drop-off fields to reduce leakage. Taxi-zone IDs are treated as numeric codes for this compact demonstration; production systems should use an appropriate categorical/geographic representation.

## Limitations

These are observational administrative records, not a random sample of all NYC travel. Taxi zones are approximate, schemas can change, and the source warns about data quality. Do not infer causality or representativeness.

The original code is MIT-licensed. Dataset access remains subject to the official source terms.
