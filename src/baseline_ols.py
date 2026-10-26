
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CATEGORICAL = ["language"]
NUMERIC = ["author_followers", "post_hour", "post_day_of_week", "hashtag_count", "duration_seconds"]

MIN_LANGUAGE_COUNT = 10


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Nema {path}. Prvo pokreni: python src/feature_engineering.py")
    return pd.read_csv(path)


def group_rare_languages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts = df["language"].value_counts()
    rare = counts[counts < MIN_LANGUAGE_COUNT].index
    df["language"] = df["language"].replace(dict.fromkeys(rare, "other_lang"))
    return df


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].copy()
    X["duration_seconds"] = X["duration_seconds"].fillna(X["duration_seconds"].median())
    dummies = pd.get_dummies(df[CATEGORICAL], drop_first=True)
    X = pd.concat([X, dummies], axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")
    return X


def check_assumptions(X_train: pd.DataFrame, residuals: pd.Series) -> None:
    print("\n--- Provera pretpostavki OLS modela ---")

    dw = durbin_watson(residuals)
    if 1.5 < dw < 2.5:
        print(f"Durbin-Watson = {dw:.3f} -> nezavisnost gresaka OK (nema znacajne autokorelacije)")
    else:
        smer = "pozitivna" if dw < 1.5 else "negativna"
        print(f"Durbin-Watson = {dw:.3f} -> {smer} autokorelacija, pretpostavka NIJE zadovoljena")

    _, bp_pvalue, _, _ = het_breuschpagan(residuals, X_train)
    if bp_pvalue > 0.05:
        print(f"Breusch-Pagan p={bp_pvalue:.4f} -> homoskedasticnost OK")
    else:
        print(f"Breusch-Pagan p={bp_pvalue:.4f} -> heteroskedasticnost, pretpostavka NIJE zadovoljena")

    _, jb_pvalue, _, _ = jarque_bera(residuals)
    if jb_pvalue > 0.05:
        print(f"Jarque-Bera p={jb_pvalue:.4f} -> reziduali su priblizno normalni")
    else:
        print(f"Jarque-Bera p={jb_pvalue:.4f} -> reziduali NISU normalno rasporedjeni")

    print("\nVIF (multikolinearnost, VIF > 10 je problematicno):")
    vif_cols = ["const"] + NUMERIC
    vif_data = X_train[vif_cols]
    for i, col in enumerate(NUMERIC, start=1):
        vif = variance_inflation_factor(vif_data.values, i)
        print(f"  {col}: VIF={vif:.2f}")


def main():
    df = load_features()
    df = group_rare_languages(df)
    X = build_design_matrix(df)
    y_log = np.log1p(df["views"])  

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    check_assumptions(X_train, model.resid)

    preds = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    print(f"\nOLS baseline (test skup): RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")
    print("Poredi sa RandomForest/XGBoost rezultatima iz 'python src/train.py'.")


if __name__ == "__main__":
    main()
