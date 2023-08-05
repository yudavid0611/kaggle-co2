import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


def score_dataset(X, y, numertic_only=True):
    model = XGBRegressor(
        max_depth=5,
        n_estimators=100,
        eval_metric='rmsle',
        random_state=1
    )

    if numertic_only:
        X = X.select_dtypes(exclude=["category", "object"])
    else:
        # Label encoding for categoricals
        for colname in X.select_dtypes(["category", "object"]):
            X[colname], _ = X[colname].factorize()
    
    score = cross_val_score(
        model, X, y, cv=3, scoring="neg_mean_absolute_error",
    )

    score = -1 * score.mean()
    score = np.sqrt(score)
    return score