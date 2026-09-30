import joblib
import pandas as pd
from load_data import load_data
from feature_engineering import create_features
from preprocess import preprocess_data

def load_model():
    model = joblib.load("models/gradient_boosting_model.pkl")
    columns = joblib.load("models/model_columns.pkl")

    return model, columns

def prepare_data():
    df = load_data()
    df = create_features(df)

    return df

def create_prediction_row(
    df,
    date,
    store_nbr,
    family,
    promotion
):
    date = pd.to_datetime(date)

    history = df[
        (df["store_nbr"] == store_nbr) &
        (df["family"] == family) &
        (df["date"] < date)
    ].copy()

    if history.empty:
        raise ValueError(
            "There is not enough historical data for this store and family."
        )

    history = history.sort_values("date")

    # get the sales history
    sales_history = history.set_index("date")["sales"]

    # create the prediction row
    row = pd.DataFrame({
        "date": [date],
        "store_nbr": [store_nbr],
        "family": [family],
        "onpromotion": [1 if promotion == "Yes" else 0]
    })

    # time features
    row["day_of_week"] = date.dayofweek
    row["day_of_month"] = date.day
    row["month"] = date.month
    row["year"] = date.year
    row["week_of_year"] = int(date.isocalendar().week)
    row["is_weekend"] = int(date.dayofweek >= 5)

    # get lag values based on the position in the time series
    sales_values = sales_history.values

    if len(sales_values) < 30:
        raise ValueError(
            "Not enough historical data to create lag features."
        )

    row["lag_1"] = sales_values[-1]
    row["lag_7"] = sales_values[-7]
    row["lag_14"] = sales_values[-14]
    row["lag_30"] = sales_values[-30]

    # rolling means
    row["rolling_mean_7"] = sales_values[-7:].mean()
    row["rolling_mean_30"] = sales_values[-30:].mean()

    return row
        
def predict_sales(model, columns, df, date, store_nbr, family, promotion):
    row = create_prediction_row(df, date, store_nbr, family, promotion)

    X, _ = preprocess_data(row)

    X = X.reindex(
        columns = columns,
        fill_value=0
    )

    prediction = model.predict(X)

    return prediction[0]