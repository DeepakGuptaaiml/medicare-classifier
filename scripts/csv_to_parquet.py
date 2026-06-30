#!/usr/bin/env python3
"""Convert claims_data.csv to parquet for Azure Blob test upload."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_data.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "claims.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert training CSV to parquet for Blob upload")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output parquet path")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {args.out}")
    print(f"  rows:    {len(df):,}")
    print(f"  columns: {len(df.columns)}")
    print()
    print("Portal upload:")
    print("  Container: medicare-training")
    print("  Blob name: latest/claims.parquet")


if __name__ == "__main__":
    main()
