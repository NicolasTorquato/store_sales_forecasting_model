import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/train.csv")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")
    # df = df.head(100000)

    start_date = "2014-04-01"
    end_date = "2014-12-31"

    df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ]

    return df