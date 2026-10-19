"""Bronze layer: load raw data as is, add tracking columns."""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

RAW = Path("data/raw/melb_data.csv")
BRONZE = Path("data/bronze/listings.parquet")


def run():
    df = pd.read_csv(RAW)

    df["listing_id"] = range(1, len(df) + 1)
    df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["_source_file"] = RAW.name

    BRONZE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE, index=False)

    print(f"Bronze: {len(df):,} rows, {df.shape[1]} columns")
    print("\nMissing values:")
    print(df.isna().sum()[lambda s: s > 0])


if __name__ == "__main__":
    run()