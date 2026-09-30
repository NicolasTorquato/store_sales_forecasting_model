def split_data(df, test_size=0.2):
    df = df.sort_values("date").copy()

    unique_dates = df["date"].sort_values().unique()

    split_index = int(len(unique_dates) * (1 - test_size))

    split_date = unique_dates[split_index]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    return train_df, test_df