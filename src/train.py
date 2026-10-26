from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

CATEGORICAL = ["market", "language"]
NUMERIC = ["author_followers", "post_hour", "post_day_of_week", "hashtag_count", "duration_seconds"]


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Nema {path}. Prvo pokreni: python src/feature_engineering.py")
    return pd.read_csv(path)


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test_log) -> float:
    preds = np.expm1(pipeline.predict(X_test))
    y_true = np.expm1(y_test_log)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    print(f"{name:12s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")
    return rmse


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()

    X = df[CATEGORICAL + NUMERIC].copy()
    X["duration_seconds"] = X["duration_seconds"].fillna(X["duration_seconds"].median())
    y_log = np.log1p(df["views"])  

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1
        ),
    }

    best_name, best_rmse, best_pipeline = None, float("inf"), None
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        rmse = evaluate(name, pipeline, X_test, y_test)
        if rmse < best_rmse:
            best_name, best_rmse, best_pipeline = name, rmse, pipeline

    out_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, out_path)
    print(f"\nNajbolji model: {best_name} -> sacuvan u {out_path}")


if __name__ == "__main__":
    main()
