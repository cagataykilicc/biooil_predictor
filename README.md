# Bio-yağ Verimi Tahmin Programı

Tarımsal biyokütle pirolizinden elde edilecek **bio-yağ verimini (%)** tahmin eden bir Python programı. Üç gradient boosting modelinin (CatBoost, LightGBM, XGBoost) ortalamasıyla tahmin yapar; ek olarak modeller arası varyansı belirsizlik göstergesi olarak sunar.

## Veri kaynağı

Eğitim verisi 504 kayıttan oluşuyor:
- **Zhao et al. (2024)** — Renewable Energy 225:120218, Tablo A.1 (163 kayıt)
- **Ortiz, M. (2021)** — Mendeley Data, DOI: 10.17632/BX88YMGBBV.1 (341 kayıt)

## Girdi parametreleri

| Parametre | Birim | Tipik aralık |
|---|---|---|
| Cellulose | % | 12 – 60 |
| Hemicellulose | % | 1 – 52 |
| Lignin | % | 1 – 48 |
| Particle size | mm | 0.2 – 3.2 |
| Pyrolysis temperature | °C | 300 – 900 |

## Kurulum

```bash
pip install pandas numpy scikit-learn catboost lightgbm xgboost joblib
```

## Kullanım

### 1) Modeli eğit (sadece bir kez)

```bash
python train_model.py --data merged_dataset.csv
```

Bu adım `models/` klasörü oluşturur:
- `catboost.pkl`, `lightgbm.pkl`, `xgboost.pkl` — eğitilmiş modeller
- `metadata.json` — özellik istatistikleri, hyperparameter'lar, performans metrikleri

### 2) Tahmin yap

**İnteraktif mod (tek tahmin):**
```bash
python predict.py
```
Program sizden parametreleri tek tek soracak ve tahmini ekrana basacak.

**Toplu mod (CSV → CSV):**
```bash
python predict.py --input my_samples.csv --output predictions.csv
```

Girdi CSV'sinde şu sütunlar bulunmalıdır:
```
Cellulose_pct,Hemicellulose_pct,Lignin_pct,ParticleSize_mm,PyrolysisTemp_C
```

## Örnek çıktı (interaktif)

```
  Bio-yağ verimi: 45.32% ± 1.18
  Aralık:         [44.05, 46.41] %

  Bireysel modeller:
    CatBoost  : 45.68%
    LightGBM  : 44.05%
    XGBoost   : 46.23%

  Eğitim verisinde gözlemlenen aralık: 13.3% – 70.4% (ortalama 42.6%)

  Model performansı (test seti üzerinde):
    R² = 0.751, RMSE = 5.25, MAE = 3.53
```

## Kısıtlamalar

- Test setinde R² ≈ 0.75 ve RMSE ≈ 5.25 puan. Yani tipik tahmin hatası ±5 puan civarında.
- Model **fixed-bed piroliz** verisi üzerinde eğitilmiş. Akışkan yatak veya mikrodalga gibi farklı reaktör tipleri için uygun değildir.
- Eğitim aralığı dışındaki girdilerde program uyarı verir; bu durumlarda tahmin güvenilirliği belirgin biçimde düşer.
- Cellulose+Hemicellulose+Lignin toplamı %100'ü aşamaz; kül ve ekstraktif maddeler bu üç bileşenin dışındadır.

## Dosya yapısı

```
biooil_predictor/
├── train_model.py        # Modeli bir kez eğitir
├── predict.py            # Tahmin programı (asıl çalıştırılan)
├── merged_dataset.csv    # Birleşik eğitim verisi
├── README.md
└── models/               # train_model.py tarafından oluşturulur
    ├── catboost.pkl
    ├── lightgbm.pkl
    ├── xgboost.pkl
    └── metadata.json
```
