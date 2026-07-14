from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


def train_model(train_df: pd.DataFrame, target_col: str, params: Dict[str, Any]):
    feature_columns = [c for c in train_df.columns if c not in [target_col, "date"]]
    X_train = train_df[feature_columns]
    y_train = train_df[target_col]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, pred)

    print(f"Training MAE: {mae}")

    # mlflow.set_experiment("sales-forecasting")
    # with mlflow.start_run(run_name="baseline") as run:
    #     mlflow.log_params(params)
    #     mlflow.log_metric("mae", mae)
    #     mlflow.sklearn.log_model(model, "model")

    # return model, run.info.run_id
    return model, mae

def load_dataset():
    df_train = pd.read_csv('data/raw/train.csv', parse_dates=['date'])
    df_test = pd.read_csv('data/raw/test.csv', parse_dates=['date'])

    # df_train.info()
    # print(df_train.head())

    target_col = "sales"

    model, _ = train_model(df_train, target_col, {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,          # random sampling of the training data
        "colsample_bynode": 0.8,   # random sampling of features for each tree node
        "random_state": 42,
    })


    return df_train, df_test

if __name__ == "__main__":
    load_dataset()