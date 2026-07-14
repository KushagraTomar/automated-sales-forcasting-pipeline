from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(train_df: pd.DataFrame, target_col: str, params: Dict[str, Any]):
    feature_columns = [c for c in train_df.columns if c not in [target_col, "date"]]
    X_train = train_df[feature_columns]
    y_train = train_df[target_col]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, pred)
    r2s = r2_score(y_train, pred)

    print(f"Training MAE: {mae}")
    print(f"Training R2 Score: {r2s}")

    # mlflow.set_experiment("sales-forecasting")
    # with mlflow.start_run(run_name="baseline") as run:
    #     mlflow.log_params(params)
    #     mlflow.log_metric("mae", mae)
    #     mlflow.sklearn.log_model(model, "model")

    # return model, run.info.run_id
    return model, r2s
