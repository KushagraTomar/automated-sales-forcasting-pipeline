from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(model, X_eval: pd.DataFrame, y_eval: pd.Series):
    preds = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, preds)
    r2s = r2_score(y_eval, preds)
    return {"mae": float(mae), "r2": float(r2s)}
