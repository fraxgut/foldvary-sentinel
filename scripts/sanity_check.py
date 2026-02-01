import sys
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "Date",
    "Event",
    "EventDetail",
    "Asset",
    "IsClusterStart",
    "DepressionScore",
    "DepressionFlag",
    "Depress_NBER_Recession",
    "Fwd_30d",
    "Fwd_90d",
    "Fwd_180d",
}

KEY_EVENTS = ["DEPRESSION_ALERT", "DEPRESSION_WATCH", "COMBO_CRISIS"]


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def warn(msg):
    print(f"WARN: {msg}")


def main():
    csv_path = Path("output/maximalist_backtest.csv")
    if not csv_path.exists():
        fail("output/maximalist_backtest.csv not found. Run `python backtest.py` first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        fail("Backtest CSV is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        fail(f"Missing required columns: {sorted(missing)}")

    if df["Event"].isna().any():
        fail("Event column contains NaN.")

    if not df["IsClusterStart"].dropna().isin([0, 1]).all():
        fail("IsClusterStart must be 0/1 only.")

    if (df["DepressionScore"] < 0).any() or (df["DepressionScore"] > 5).any():
        fail("DepressionScore outside expected range 0-5.")

    expected_flag = (df["DepressionScore"] >= 3).astype(int)
    if not (df["DepressionFlag"] == expected_flag).all():
        fail("DepressionFlag inconsistent with DepressionScore >= 3 rule.")

    for event in KEY_EVENTS:
        if not (df["Event"] == event).any():
            warn(f"{event} not present in backtest output.")

    print("PASS: sanity checks completed.")


if __name__ == "__main__":
    main()
