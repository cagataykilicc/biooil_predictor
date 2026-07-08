"""
train_model.py
==============
Tarımsal biyokütle pirolizinden bio-oil verimi tahmin modelini eğitir.

Birleşik veri kümesi (Zhao 2024 Tablo A.1 + Ortiz 2021) üzerinde
11 farklı regresyon modelini eğitir, modelleri models/ klasörüne kaydeder.

Kullanım:
    python train_model.py --data merged_dataset.csv
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
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

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


def get_models():
    """11 modeli belirtilen hiperparametrelerle başlatır."""
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "svr": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
        ),
        "xgboost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }


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
        "--data",
        required=True,
        help="Birleşik veri kümesinin CSV yolu (örn. merged_dataset.csv)",
    )
    parser.add_argument(
        "--out",
        default="models",
        help="Modellerin kaydedileceği klasör (varsayılan: models/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Veri yükleniyor: {args.data}")
    df = pd.read_csv(args.data)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV'de eksik sütun(lar): {missing_cols}")

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"    Kullanılacak satır sayısı: {len(df)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[2/4] Modeller eğitiliyor ve Cross-Validation yapılıyor...")
    models = get_models()
    trained_models = {}
    model_metrics = {}
    model_cv_scores = {}

    for name, model in models.items():
        print(f"  -> Eğitiliyor: {name}")
        # Cross-validation R2 skoru
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_r2 = float(np.mean(cv_scores))
        model_cv_scores[name] = cv_r2

        # Modeli eğit
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Test seti değerlendirmesi
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        model_metrics[name] = metrics
        print(f"     CV R2: {cv_r2:.4f} | Test R2: {metrics['r2_test']:.4f}")

    print("\n[3/4] Modeller kaydediliyor...")
    for name, model in trained_models.items():
        joblib.dump(model, out_dir / f"{name}.pkl")

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

    # Metadata oluştur
    model_meta = {}
    for name in models.keys():
        # Get parameters safely
        params = trained_models[name].get_params()
        # Filter parameters that are not JSON serializable if any
        serializable_params = {}
        for k, v in params.items():
            try:
                json.dumps({k: v})
                serializable_params[k] = v
            except TypeError:
                serializable_params[k] = str(v)

        model_meta[name] = {
            "best_params": serializable_params,
            "cv_r2": model_cv_scores[name],
            **model_metrics[name],
        }

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_stats": feature_stats,
        "target_stats": target_stats,
        "models": model_meta,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[4/4] Tamamlandı. Çıktılar: {out_dir.resolve()}")
    print("\n" + "=" * 50)
    print("  MODEL PERFORMANS ÖZETİ (R2)")
    print("=" * 50)
    summary_df = pd.DataFrame({
        "Model": list(models.keys()),
        "CV R2": [model_cv_scores[m] for m in models.keys()],
        "Test R2": [model_metrics[m]["r2_test"] for m in models.keys()],
        "Test RMSE": [model_metrics[m]["rmse_test"] for m in models.keys()],
        "Test MAE": [model_metrics[m]["mae_test"] for m in models.keys()],
    })
    summary_df = summary_df.sort_values(by="Test R2", ascending=False)
    for idx, row in summary_df.iterrows():
        print(f"  {row['Model']:<20} -> CV R2={row['CV R2']:.4f} | Test R2={row['Test R2']:.4f} | RMSE={row['Test RMSE']:.2f}")


if __name__ == "__main__":
    main()
