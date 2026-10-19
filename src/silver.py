"""Silver layer: clean, add features and dealbreaker flags."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

BRONZE = Path("data/bronze/listings.parquet")
SILVER = Path("data/silver/listings.parquet")
CONFIG = Path("config/dealbreakers.yaml")

RENAME = {
    "listing_id": "listing_id",
    "Suburb": "suburb",
    "Address": "address",
    "Rooms": "rooms",
    "Type": "type",
    "Price": "price",
    "Date": "sale_date",
    "Distance": "distance",
    "Bathroom": "bathroom",
    "Car": "car",
    "Landsize": "landsize",
    "BuildingArea": "building_area",
    "YearBuilt": "year_built",
    "Regionname": "region",
    "Lattitude": "lat",
    "Longtitude": "lng",
}


def load_categories():
    with open(CONFIG) as f:
        return yaml.safe_load(f)["categories"]


def clean(df):
    df = df.rename(columns=RENAME)[list(RENAME.values())].copy()
    df = df.dropna(subset=["price"])

    # Fill car and bathroom with suburb median, then overall median
    for col in ["car", "bathroom"]:
        df[col] = df[col].fillna(df.groupby("suburb")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # Land size 0 means unknown, not zero land
    df["landsize"] = df["landsize"].replace(0, np.nan)

    # Year built: remove impossible values, flag missing, never guess
    df.loc[df["year_built"] < 1800, "year_built"] = np.nan
    df["year_built_missing"] = df["year_built"].isna().astype(int)

    df["sale_date"] = pd.to_datetime(df["sale_date"], dayfirst=True)
    df["age"] = df["sale_date"].dt.year - df["year_built"]
    return df


def add_features(df):
    for t in ["h", "u", "t"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    return df


def add_dealbreaker_flags(df, categories):
    for name, cat in categories.items():
        df[f"db_{name}"] = df.eval(cat["rule"]).fillna(False).astype(int)
    return df


def run():
    df = pd.read_parquet(BRONZE)
    df = clean(df)
    df = add_features(df)
    df = add_dealbreaker_flags(df, load_categories())

    SILVER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER, index=False)

    print(f"Silver: {len(df):,} rows, {df.shape[1]} columns")
    print("\nDealbreaker flag rates:")
    flags = [c for c in df.columns if c.startswith("db_")]
    print((df[flags].mean() * 100).round(1).astype(str) + "%")


if __name__ == "__main__":
    run()