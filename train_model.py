"""
train_model.py
==============
Tarımsal biyokütle pirolizinden bio-oil verimi tahmin modelini eğitir.

Birleşik veri kümesi (Zhao 2024 Tablo A.1 + Ortiz 2021) üzerinde
CatBoost, LightGBM ve XGBoost'u eğitir, hyperparameter tuning yapar,
ve modelleri models/ klasörüne kaydeder.

Kullanım:
    python train_model.py --data merged_dataset.csv

Bu betiği SADECE BİR KEZ çalıştırmanız yeterli. Ardından predict.py'yi
kullanarak istediğiniz kadar tahmin yapabilirsiniz.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------- Sabitler ----------
FEATURES = [
    "Cellulose_pct",
    "Hemicellulose_pct",
    "Lignin_pct",
    "ParticleSize_mm",
    "PyrolysisTemp_C",
]
TARGET = "BioOilYield_pct"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def tune_catboost(X, y, cv):
    grid = {
        "iterations": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5, 7],
    }
    base = CatBoostRegressor(
        random_state=RANDOM_STATE, verbose=0, allow_writing_files=False
    )
    search = RandomizedSearchCV(
        base, grid, n_iter=30, cv=cv,
        scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(X, y, cv):
    grid = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8, -1],
        "min_child_samples": [5, 10, 20],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
    }
    base = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    search = RandomizedSearchCV(
        base, grid, n_iter=60, cv=cv,
        scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgboost(X, y, cv):
    grid = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.5, 1.0],
    }
    base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(
        base, grid, n_iter=60, cv=cv,
        scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(model, X_train, X_test, y_train, y_test):
    yp_tr = model.predict(X_train)
    yp_te = model.predict(X_test)
    return {
        "r2_train": float(r2_score(y_train, yp_tr)),
        "r2_test": float(r2_score(y_test, yp_te)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, yp_te))),
        "mae_test": float(mean_absolute_error(y_test, yp_te)),
    }


def main():
    parser = argparse.ArgumentParser(description="Bio-oil yield model trainer")
    parser.add_argument(
        "--data", required=True,
        help="Birleşik veri kümesinin CSV yolu (örn. merged_dataset.csv)",
    )
    parser.add_argument(
        "--out", default="models",
        help="Modellerin kaydedileceği klasör (varsayılan: models/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Veri yükleniyor: {args.data}")
    df = pd.read_csv(args.data)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV'de eksik sütun(lar): {missing_cols}")

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"    Kullanılacak satır sayısı: {len(df)}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    print("[2/5] CatBoost tuning...")
    cb_model, cb_params, cb_cv = tune_catboost(X_train, y_train, cv)
    cb_metrics = evaluate(cb_model, X_train, X_test, y_train, y_test)
    print(f"    CV R2: {cb_cv:.4f} | Test R2: {cb_metrics['r2_test']:.4f}")

    print("[3/5] LightGBM tuning...")
    lgb_model, lgb_params, lgb_cv = tune_lightgbm(X_train, y_train, cv)
    lgb_metrics = evaluate(lgb_model, X_train, X_test, y_train, y_test)
    print(f"    CV R2: {lgb_cv:.4f} | Test R2: {lgb_metrics['r2_test']:.4f}")

    print("[4/5] XGBoost tuning...")
    xgb_model, xgb_params, xgb_cv = tune_xgboost(X_train, y_train, cv)
    xgb_metrics = evaluate(xgb_model, X_train, X_test, y_train, y_test)
    print(f"    CV R2: {xgb_cv:.4f} | Test R2: {xgb_metrics['r2_test']:.4f}")

    print("[5/5] Modeller ve metadata kaydediliyor...")
    joblib.dump(cb_model, out_dir / "catboost.pkl")
    joblib.dump(lgb_model, out_dir / "lightgbm.pkl")
    joblib.dump(xgb_model, out_dir / "xgboost.pkl")

    # Özellik istatistikleri (predict.py extrapolation uyarısı için)
    feature_stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "p05": float(df[col].quantile(0.05)),
            "p95": float(df[col].quantile(0.95)),
        }
        for col in FEATURES
    }
    target_stats = {
        "min": float(df[TARGET].min()),
        "max": float(df[TARGET].max()),
        "mean": float(df[TARGET].mean()),
    }

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_stats": feature_stats,
        "target_stats": target_stats,
        "models": {
            "catboost": {
                "best_params": cb_params,
                "cv_r2": float(cb_cv),
                **cb_metrics,
            },
            "lightgbm": {
                "best_params": lgb_params,
                "cv_r2": float(lgb_cv),
                **lgb_metrics,
            },
            "xgboost": {
                "best_params": xgb_params,
                "cv_r2": float(xgb_cv),
                **xgb_metrics,
            },
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Tamamlandı. Çıktılar: {out_dir.resolve()}")
    print("\nÖzet:")
    print(f"  CatBoost  -> Test R2={cb_metrics['r2_test']:.4f}  RMSE={cb_metrics['rmse_test']:.2f}")
    print(f"  LightGBM  -> Test R2={lgb_metrics['r2_test']:.4f}  RMSE={lgb_metrics['rmse_test']:.2f}")
    print(f"  XGBoost   -> Test R2={xgb_metrics['r2_test']:.4f}  RMSE={xgb_metrics['rmse_test']:.2f}")


if __name__ == "__main__":
    main()
