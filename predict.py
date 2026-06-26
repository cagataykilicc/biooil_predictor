"""
predict.py
==========
Eğitilmiş modelleri kullanarak tarımsal biyokütle pirolizinden bio-oil verimi
tahmini yapar.

İki kullanım modu:

1) İnteraktif (tek tahmin):
       python predict.py

2) Toplu (CSV → CSV):
       python predict.py --input my_samples.csv --output predictions.csv

   Girdi CSV'sinde şu sütunlar bulunmalıdır:
       Cellulose_pct, Hemicellulose_pct, Lignin_pct,
       ParticleSize_mm, PyrolysisTemp_C
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path(__file__).parent / "models"


# ---------- Model yükleme ----------
def load_artifacts(models_dir: Path):
    """Eğitilmiş modelleri ve metadata'yı yükler."""
    if not models_dir.exists():
        sys.exit(
            f"HATA: '{models_dir}' klasörü bulunamadı.\n"
            f"Önce şu komutu çalıştırın:\n"
            f"    python train_model.py --data merged_dataset.csv"
        )
    try:
        cb = joblib.load(models_dir / "catboost.pkl")
        lgb = joblib.load(models_dir / "lightgbm.pkl")
        xgb = joblib.load(models_dir / "xgboost.pkl")
        with open(models_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        sys.exit(f"HATA: Eksik model dosyası: {e.filename}")
    return {"catboost": cb, "lightgbm": lgb, "xgboost": xgb}, metadata


# ---------- Girdi doğrulama ----------
def validate_inputs(values: dict, metadata: dict) -> list:
    """Girdileri eğitim aralığıyla karşılaştırır, uyarı listesi döner."""
    warnings = []
    stats = metadata["feature_stats"]

    # Fiziksel kontroller
    cel = values["Cellulose_pct"]
    hem = values["Hemicellulose_pct"]
    lig = values["Lignin_pct"]
    total = cel + hem + lig

    for name, v in [("Cellulose", cel), ("Hemicellulose", hem), ("Lignin", lig)]:
        if v < 0:
            warnings.append(f"[UYARI] {name} negatif olamaz: {v}")
        elif v > 100:
            warnings.append(f"[UYARI] {name} %100'den büyük olamaz: {v}")

    if total > 100:
        warnings.append(
            f"[UYARI] Cel+Hem+Lig toplamı {total:.1f}% (>100). "
            f"Fiziksel olarak imkansız; girdi değerlerinizi kontrol edin."
        )
    elif total < 50:
        warnings.append(
            f"[UYARI] Cel+Hem+Lig toplamı yalnızca {total:.1f}%. "
            f"Bu çok düşük; tarımsal biyokütlede genelde 70-95% beklenir."
        )

    if values["ParticleSize_mm"] <= 0:
        warnings.append(f"[UYARI] Particle size pozitif olmalı: {values['ParticleSize_mm']}")

    if values["PyrolysisTemp_C"] < 200 or values["PyrolysisTemp_C"] > 1200:
        warnings.append(
            f"[UYARI] Piroliz sıcaklığı tipik aralığın dışında: {values['PyrolysisTemp_C']}°C "
            f"(beklenen 300-900°C)"
        )

    # Eğitim verisi aralığı dışı (extrapolation) uyarısı
    for col in ["Cellulose_pct", "Hemicellulose_pct", "Lignin_pct",
                "ParticleSize_mm", "PyrolysisTemp_C"]:
        v = values[col]
        s = stats[col]
        if v < s["min"] or v > s["max"]:
            warnings.append(
                f"[UYARI] {col} = {v} -> eğitim aralığının DIŞINDA "
                f"[{s['min']:.2f}, {s['max']:.2f}]. Tahmin güvenilirliği düşer."
            )
        elif v < s["p05"] or v > s["p95"]:
            warnings.append(
                f"[BILGI] {col} = {v} -> eğitim verisinin uç bölgesinde "
                f"(p05={s['p05']:.2f}, p95={s['p95']:.2f}). Tahmin daha az güvenilir."
            )
    return warnings


# ---------- Tahmin ----------
def predict_ensemble(models, features_array, feature_names):
    """3 modelle tahmin yapar; ortalama, std, min, max döner.

    LightGBM/XGBoost feature_names ile fit edildiğinden, tahmin için
    numpy array yerine DataFrame geçiriyoruz (sklearn UserWarning'ini önler).
    """
    X_df = pd.DataFrame(features_array, columns=feature_names)
    preds = np.array([
        models["catboost"].predict(X_df),
        models["lightgbm"].predict(X_df),
        models["xgboost"].predict(X_df),
    ])  # shape: (3, n_samples)
    return {
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "min": preds.min(axis=0),
        "max": preds.max(axis=0),
        "catboost": preds[0],
        "lightgbm": preds[1],
        "xgboost": preds[2],
    }


# ---------- İnteraktif mod ----------
def prompt_float(prompt: str, default=None) -> float:
    """Kullanıcıdan float değer alır; geçersiz girdiye karşı korumalı."""
    while True:
        suffix = f" [varsayılan: {default}]" if default is not None else ""
        s = input(f"  {prompt}{suffix}: ").strip().replace(",", ".")
        if not s and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("    [HATA] Geçersiz sayı. Tekrar deneyin.")


def interactive_mode(models, metadata):
    print("=" * 70)
    print("  TARIMSAL BİYOKÜTLE PİROLİZİ — BİO-YAĞ VERİMİ TAHMİN PROGRAMI")
    print("=" * 70)
    print("\nLütfen biyokütle özelliklerini ve piroliz koşullarını girin.")
    print("(Ondalık ayraç olarak nokta veya virgül kullanabilirsiniz.)\n")

    print("— Biyokütle bileşimi —")
    cel = prompt_float("Cellulose içeriği (%)")
    hem = prompt_float("Hemicellulose içeriği (%)")
    lig = prompt_float("Lignin içeriği (%)")

    print("\n— Piroliz koşulları —")
    ps = prompt_float("Partikül boyutu (mm)")
    pt = prompt_float("Piroliz sıcaklığı (°C)")

    values = {
        "Cellulose_pct": cel,
        "Hemicellulose_pct": hem,
        "Lignin_pct": lig,
        "ParticleSize_mm": ps,
        "PyrolysisTemp_C": pt,
    }

    warnings_list = validate_inputs(values, metadata)
    if warnings_list:
        print("\n— Uyarılar —")
        for w in warnings_list:
            print(f"  {w}")

    X = np.array([[values[f] for f in metadata["features"]]])
    res = predict_ensemble(models, X, metadata["features"])

    print("\n" + "=" * 70)
    print("  TAHMİN SONUCU")
    print("=" * 70)
    print(f"\n  Bio-yağ verimi: {res['mean'][0]:.2f}% +/- {res['std'][0]:.2f}")
    print(f"  Aralık:         [{res['min'][0]:.2f}, {res['max'][0]:.2f}] %")
    print(f"\n  Bireysel modeller:")
    print(f"    CatBoost  : {res['catboost'][0]:.2f}%")
    print(f"    LightGBM  : {res['lightgbm'][0]:.2f}%")
    print(f"    XGBoost   : {res['xgboost'][0]:.2f}%")

    # Bağlam bilgisi
    t = metadata["target_stats"]
    print(f"\n  Eğitim verisinde gözlemlenen aralık: "
          f"{t['min']:.1f}% – {t['max']:.1f}% (ortalama {t['mean']:.1f}%)")

    # Performans bilgisi
    cb = metadata["models"]["catboost"]
    print(f"\n  Model performansı (test seti üzerinde):")
    print(f"    R2 = {cb['r2_test']:.3f}, RMSE = {cb['rmse_test']:.2f}, "
          f"MAE = {cb['mae_test']:.2f}")
    print()


# ---------- Toplu (CSV) mod ----------
def batch_mode(input_csv: str, output_csv: str, models, metadata):
    df = pd.read_csv(input_csv)
    required = metadata["features"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"HATA: Girdi CSV'sinde şu sütunlar eksik: {missing}\n"
            f"Gerekli sütunlar: {required}"
        )

    print(f"Girdi: {input_csv} ({len(df)} satır)")

    # Eksik değer satırlarını ayıkla
    valid = df[required].notna().all(axis=1)
    if (~valid).any():
        print(f"  {(~valid).sum()} satır eksik değer içerdiğinden atlandı.")
    df_valid = df[valid].copy()

    X = df_valid[required].values
    res = predict_ensemble(models, X, required)

    df_valid["Predicted_BioOilYield_pct"] = res["mean"]
    df_valid["Prediction_Std"] = res["std"]
    df_valid["Prediction_Min"] = res["min"]
    df_valid["Prediction_Max"] = res["max"]
    df_valid["CatBoost_Pred"] = res["catboost"]
    df_valid["LightGBM_Pred"] = res["lightgbm"]
    df_valid["XGBoost_Pred"] = res["xgboost"]

    # Satır bazında uyarılar
    warning_strings = []
    for _, row in df_valid.iterrows():
        ws = validate_inputs(
            {f: row[f] for f in required}, metadata
        )
        warning_strings.append(" | ".join(ws) if ws else "")
    df_valid["Warnings"] = warning_strings

    df_valid.to_csv(output_csv, index=False)
    print(f"Çıktı: {output_csv}")
    print(
        f"\nÖzet - ortalama tahmin: {res['mean'].mean():.2f}%  "
        f"(min {res['mean'].min():.2f}%, max {res['mean'].max():.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bio-yağ verimi tahmin programı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", help="Toplu mod için girdi CSV dosyası"
    )
    parser.add_argument(
        "--output", "-o", default="predictions.csv",
        help="Toplu mod için çıktı CSV dosyası (varsayılan: predictions.csv)",
    )
    parser.add_argument(
        "--models-dir", default=str(DEFAULT_MODELS_DIR),
        help=f"Eğitilmiş model klasörü (varsayılan: {DEFAULT_MODELS_DIR})",
    )
    args = parser.parse_args()

    models, metadata = load_artifacts(Path(args.models_dir))

    if args.input:
        batch_mode(args.input, args.output, models, metadata)
    else:
        interactive_mode(models, metadata)


if __name__ == "__main__":
    main()
