import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame, lags=None, windows=None) -> pd.DataFrame:
    if lags is None:
        lags = [7, 14, 28, 365]
    if windows is None:
        windows = [7, 30]

    df = df.sort_values(["store", "item", "date"]).copy()
    df = add_time_features(df)

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(["store", "item"])["sales"].shift(lag)
        )

    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store", "item"])["sales"]
                .transform(lambda s: s.rolling(window, min_periods=window)
                    .mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(["store", "item"])["sales"]
                .transform(lambda s: s.rolling(window, min_periods=window)
                    .std())
        )

    return df

if __name__ == "__main__":
    df_train = pd.read_csv('data/raw/train.csv', parse_dates=['date'])
    df_train = add_lag_and_rolling_features(df_train)
    print(df_train.head(5))

