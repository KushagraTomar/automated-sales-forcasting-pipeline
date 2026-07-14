from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# mlflow.sklearn.autolog()

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]):

    model = xgb.XGBRegressor(random_state=42)

    # grid search with 5-fold cross-validation
    grid_search = GridSearchCV(model, params, cv = 5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # best parameters and best score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    pred = best_model.predict(X_train)
    mae = mean_absolute_error(y_train, pred)
    r2s = r2_score(y_train, pred)

    print(f"Training MAE: {mae}")
    print(f"Training R2 Score: {r2s}")

    mlflow.set_experiment("sales-forecasting")
    run_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # grid_params = {f"grid_{k}": str(v) for k, v in params.items()}
        # mlflow.log_params(grid_params)
        # Log the best parameters found
        mlflow.log_params(best_params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2s)
        mlflow.xgboost.log_model(best_model, "model")

    return best_model, run.info.run_id
