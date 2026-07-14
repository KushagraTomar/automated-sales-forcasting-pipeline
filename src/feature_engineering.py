import pandas as pd


def add_time_features(df: pd.DataFrame):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame, lags=None, avgs=None):
    if lags is None:
        lags = [7, 28, 365]
    if avgs is None:
        avgs = [7, 28]

    df = df.sort_values(["store", "item", "date"]).copy()
    df = add_time_features(df)

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(["store", "item"])["sales"].shift(lag)
        )

    for avg in avgs:
        df[f"rolling_mean_{avg}"] = (
            df.groupby(["store", "item"])["sales"]
                .transform(lambda s: s.rolling(avg, min_periods=avg)
                    .mean())
        )
        df[f"rolling_std_{avg}"] = (
            df.groupby(["store", "item"])["sales"]
                .transform(lambda s: s.rolling(avg, min_periods=avg)
                    .std())
        )

    return df

