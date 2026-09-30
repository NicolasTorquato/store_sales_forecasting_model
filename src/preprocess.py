import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # encode categorical features
    df = pd.get_dummies(
        df,
        columns=["family"],
        dtype=int
    )

    # remove columns that should not be used by the model
    df = df.drop(columns=["date"])

    # separate features and target
    if "sales" in df.columns:
        X = df.drop(columns=["sales"])
        y = df["sales"]
    else:
        X = df
        y = None

    return X, y