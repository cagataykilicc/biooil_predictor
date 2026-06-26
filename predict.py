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


# ---------- Hazır Biyokütle Kütüphanesi & Optimizasyon ----------
FEEDSTOCK_PRESETS = {
    "1": {"name": "Pirinc sapi (Rice straw)", "Cellulose_pct": 36.5, "Hemicellulose_pct": 24.0, "Lignin_pct": 15.4},
    "2": {"name": "Bugday sapi (Wheat straw)", "Cellulose_pct": 38.6, "Hemicellulose_pct": 28.5, "Lignin_pct": 15.0},
    "3": {"name": "Misir kocani (Corn cob)", "Cellulose_pct": 35.0, "Hemicellulose_pct": 28.0, "Lignin_pct": 15.0},
    "4": {"name": "Findik kabugu (Hazelnut shell)", "Cellulose_pct": 26.8, "Hemicellulose_pct": 30.4, "Lignin_pct": 42.9},
    "5": {"name": "Pamuk sapi (Cotton stalk)", "Cellulose_pct": 45.0, "Hemicellulose_pct": 21.0, "Lignin_pct": 30.0},
    "6": {"name": "Ayciçek sapi (Sunflower stalk)", "Cellulose_pct": 42.1, "Hemicellulose_pct": 29.7, "Lignin_pct": 13.4},
}


def optimize_pyrolysis_temp(models, base_values, feature_names):
    """Mevcut girdiler için en yüksek bio-yağ verimini sağlayan piroliz sıcaklığını simüle eder."""
    temp_range = list(range(300, 901, 5))
    test_rows = []
    for t in temp_range:
        row = base_values.copy()
        row["PyrolysisTemp_C"] = t
        test_rows.append(row)
    
    test_df = pd.DataFrame(test_rows)
    res = predict_ensemble(models, test_df[feature_names].values, feature_names)
    mean_preds = res["mean"]
    
    max_idx = np.argmax(mean_preds)
    best_temp = temp_range[max_idx]
    best_yield = mean_preds[max_idx]
    
    return int(best_temp), float(best_yield)


def save_prediction_to_history(sample_name, values, res, opt_temp, opt_yield):
    """İnteraktif modda yapılan tahmini predictions_history.csv dosyasına kaydeder."""
    from datetime import datetime
    history_file = Path("predictions_history.csv")
    
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Biomass_name": sample_name,
        "Cellulose_pct": values["Cellulose_pct"],
        "Hemicellulose_pct": values["Hemicellulose_pct"],
        "Lignin_pct": values["Lignin_pct"],
        "ParticleSize_mm": values["ParticleSize_mm"],
        "PyrolysisTemp_C": values["PyrolysisTemp_C"],
        "Predicted_BioOilYield_pct": float(res["mean"][0]),
        "Prediction_Std": float(res["std"][0]),
        "Prediction_Min": float(res["min"][0]),
        "Prediction_Max": float(res["max"][0]),
        "CatBoost_Pred": float(res["catboost"][0]),
        "LightGBM_Pred": float(res["lightgbm"][0]),
        "XGBoost_Pred": float(res["xgboost"][0]),
        "Optimum_PyrolysisTemp_C": opt_temp,
        "Max_Predicted_BioOilYield_pct": opt_yield
    }
    
    df_new = pd.DataFrame([new_row])
    
    if history_file.exists():
        try:
            df_hist = pd.read_csv(history_file)
            df_combined = pd.concat([df_hist, df_new], ignore_index=True)
            df_combined.to_csv(history_file, index=False)
        except Exception:
            df_new.to_csv(history_file, index=False)
    else:
        df_new.to_csv(history_file, index=False)
        
    print(f"  >> Tahmin geçmişi başarıyla kaydedildi: {history_file.resolve()}")


# ---------- İnteraktif mod ----------
class AbortException(Exception):
    """Kullanıcı işlemi iptal etmek istediğinde fırlatılan özel hata."""
    pass


def prompt_float(prompt: str, default=None) -> float:
    """Kullanıcıdan float değer alır; geçersiz girdiye karşı korumalı."""
    while True:
        suffix = f" [varsayılan: {default}]" if default is not None else ""
        s = input(f"  {prompt}{suffix}: ").strip().replace(",", ".")
        if s.lower() in ["q", "iptal", "exit", "cancel"]:
            raise AbortException()
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
    print("(Herhangi bir adımda 'q', 'iptal' veya 'exit' yazarak işlemi iptal edebilirsiniz.)")
    print("(Ondalık ayraç olarak nokta veya virgül kullanabilirsiniz.)\n")

    try:
        # Biyokütle / Numune ismi alma
        sample_name = input("Lütfen biyokütle / numune adını girin (Örn: Çam Talaşı): ").strip()
        if sample_name.lower() in ["q", "iptal", "exit", "cancel"]:
            raise AbortException()
        if not sample_name:
            sample_name = "Bilinmeyen Biyokütle"

        # Hazır biyokütle kütüphanesi seçimi
        print("\nBiyokütle bileşim veri kaynağını seçin:")
        print("  [0] Kendi değerlerimi girmek istiyorum (Manuel)")
        for key, val in FEEDSTOCK_PRESETS.items():
            print(f"  [{key}] {val['name']} (Cel: {val['Cellulose_pct']}%, Hem: {val['Hemicellulose_pct']}%, Lig: {val['Lignin_pct']}%)")
        
        choice = input("\nSeçiminiz [varsayılan: 0]: ").strip()
        if choice.lower() in ["q", "iptal", "exit", "cancel"]:
            raise AbortException()
        
        cel, hem, lig = None, None, None
        if choice in FEEDSTOCK_PRESETS:
            preset = FEEDSTOCK_PRESETS[choice]
            print(f"\n  >> Hazır Biyokütle Seçildi: {preset['name']}")
            cel = preset["Cellulose_pct"]
            hem = preset["Hemicellulose_pct"]
            lig = preset["Lignin_pct"]
        else:
            print("\n— Biyokütle bileşimi (Manuel) —")
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
        print(f"\n  Biyokütle Adı:  {sample_name}")
        print(f"  Bio-yağ verimi: {res['mean'][0]:.2f}% +/- {res['std'][0]:.2f}")
        print(f"  Aralık:         [{res['min'][0]:.2f}, {res['max'][0]:.2f}] %")
        print(f"\n  Bireysel modeller:")
        print(f"    CatBoost  : {res['catboost'][0]:.2f}%")
        print(f"    LightGBM  : {res['lightgbm'][0]:.2f}%")
        print(f"    XGBoost   : {res['xgboost'][0]:.2f}%")

        # Sıcaklık Optimizasyon Motoru
        opt_temp, opt_yield = optimize_pyrolysis_temp(models, values, metadata["features"])
        yield_diff = opt_yield - res['mean'][0]
        
        print("\n" + "=" * 70)
        print("  SICAKLIK OPTİMİZASYON TAVSİYESİ")
        print("=" * 70)
        print(f"\n  Girdiğiniz bileşim ve partikül boyutu ({ps} mm) için:")
        print(f"    En yüksek verimi sağlayan optimum sıcaklık: {opt_temp}°C")
        print(f"    Bu sıcaklıktaki tahmini bio-yağ verimi:    %{opt_yield:.2f}")
        if abs(opt_temp - pt) <= 5:
            print(f"    >> Harika! Zaten optimum sıcaklığa ({pt}°C) çok yakın çalışıyorsunuz.")
        else:
            print(f"    >> Tavsiye: Sıcaklığı {pt}°C değerinden {opt_temp}°C değerine ayarlamak,")
            print(f"       verimi yaklaşık %{abs(yield_diff):.2f} oranında değiştirebilir.")

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
        
        # Tahmin geçmişini kaydet
        save_prediction_to_history(sample_name, values, res, opt_temp, opt_yield)
        
    except AbortException:
        print("\n[BILGI] İşlem kullanıcı tarafından iptal edildi.")
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

    # Satır bazında uyarılar ve optimizasyon
    warning_strings = []
    opt_temps = []
    opt_yields = []
    for _, row in df_valid.iterrows():
        row_dict = {f: row[f] for f in required}
        ws = validate_inputs(row_dict, metadata)
        warning_strings.append(" | ".join(ws) if ws else "")
        
        # Optimum koşul hesabı
        opt_t, opt_y = optimize_pyrolysis_temp(models, row_dict, required)
        opt_temps.append(opt_t)
        opt_yields.append(opt_y)

    df_valid["Warnings"] = warning_strings
    df_valid["Optimum_PyrolysisTemp_C"] = opt_temps
    df_valid["Max_Predicted_BioOilYield_pct"] = opt_yields

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
