import pandas as pd

def create_time_features(df):
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df

def create_lag_features(df):
    df = df.copy()

    group = df.groupby(["store_nbr", "family"])["sales"]
    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_30"] = group.shift(30)

    return df

def create_rolling_features(df):
    df = df.copy()

    df["rolling_mean_7"] = (
        df.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df["rolling_mean_30"] = (
        df.groupby(["store_nbr", "family"])["sales"].transform(lambda x: x.shift(1).rolling(30).mean())
    )

    return df

def create_features(df):
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    return df