# 🌱 Tarımsal Biyokütle Pirolizi — Bio-Yağ Verimi Tahmin Programı (11 Algoritma)

Tarımsal biyokütle pirolizinden elde edilecek **bio-yağ verimini (%)** tahmin eden ve piroliz sıcaklığını optimize eden interaktif bir yapay zeka aracıdır. 

Tez çalışmanız kapsamında **11 farklı makine öğrenmesi algoritmasının** (Doğrusal Regresyon, Ridge, k-NN, SVR, Karar Ağacı, Rastgele Orman, Ekstra Ağaçlar, Gradyan Artırma, XGBoost, LightGBM, CatBoost) karşılaştırmalı analizi ve bu modellerin topluluk (ensemble) tahminiyle çalışır. Modeller arası varyansı tahmin belirsizliği (sapma) olarak sunar.

---

## 🚀 Öne Çıkan Özellikler

*   **📈 11 Algoritmalı Tahmin Sistemi:** Sabit hiperparametrelerle eğitilen 11 farklı regresyon modelinin tahmin gücü karşılaştırılır ve ensemble tahminiyle yüksek kararlılık sağlanır.
*   **🌐 İnteraktif Web Dashboard (Streamlit):** Girdileri sliders yardımıyla kontrol edebileceğiniz, anlık tahmin grafiklerini ve sıcaklık-verim analizlerini izleyebileceğiniz modern bir web uygulaması (`app.py`).
*   **🌾 Hazır Biyokütle Kütüphanesi (Presets):** Yaygın tarımsal biyokütlelerin (Pirinç Sapı, Buğday Sapı, Mısır Koçanı, Fındık Kabuğu vb.) kimyasal bileşen yüzdeleri sisteme tanımlanmıştır. Tek tıkla otomatik doldurulabilir.
*   **🎯 Sıcaklık Optimizasyon Motoru:** Biyokütle bileşimini sabit tutarak 300°C - 900°C arasındaki tüm sıcaklıkları simüle eder ve en yüksek verimi alabileceğiniz **optimum piroliz sıcaklığını** önerir.
*   **💾 Tahmin Geçmişi Kaydedici (History Logging):** İnteraktif modda yaptığınız tüm tahminler anlık tarih/saat damgası ile `predictions_history.csv` dosyasına otomatik olarak eklenir.
*   **🛑 Kolay İptal (CLI):** Komut satırında veri girerken herhangi bir adımda `q`, `iptal` veya `exit` yazarak işlemi güvenle sonlandırabilirsiniz.

---

## 🛠️ Kurulum

Gerekli tüm kütüphaneleri sanal ortamınıza yüklemek için:

```bash
pip install -r requirements.txt
```

---

## 💻 Kullanım Yöntemleri

### Yöntem 1: İnteraktif Web Dashboard (Önerilen)

Sürükle-bırak kontrol araçları ve dinamik grafikler içeren web arayüzünü başlatmak için:

```bash
streamlit run app.py
```
Arayüz tarayıcınızda otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### Yöntem 2: Komut Satırı İnteraktif Mod (Tekil Tahmin)

Konsol üzerinden adım adım numune adı, hazır biyokütle seçimi ve sıcaklık değerlerini girerek tahmin yapmak için:

```bash
python predict.py
```
*Herhangi bir adımda çıkış yapmak için `q` yazabilirsiniz. Tahminleriniz `predictions_history.csv` dosyasına kaydedilecektir.*

### Yöntem 3: Toplu Tahmin Modu (CSV → CSV)

Elindeki çok sayıda numunenin tahminlerini ve optimum sıcaklık hesaplamalarını toplu olarak yapıp kaydetmek için:

```bash
python predict.py --input test_samples.csv --output predictions.csv
```
*Bu modda çıktı dosyasına (`predictions.csv`) her numune için hesaplanan optimum sıcaklık (`Optimum_PyrolysisTemp_C`) ve maksimum verim (`Max_Predicted_BioOilYield_pct`) sütunları eklenir.*

---

## 📊 Akademik Grafikleri Oluşturma (Korelasyon ve Regresyon)

Çalışmanızda, tez raporunuzda veya sunumlarınızda kullanmak üzere yayın kalitesinde (300 DPI) akademik grafikler üretmek için hazırlanan `generate_plots.py` betiğini çalıştırabilirsiniz:

```bash
python generate_plots.py
```

Bu betik çalıştırıldığında proje klasörünüzde şu 3 görsel üretilecektir:
1. **`correlation_matrix.png`**: Girdi özellikleri ve hedef biyo-yağ verimi arasındaki doğrusal Pearson korelasyon katsayılarını gösteren alt üçgensel ısı haritası (Heatmap).
2. **`regression_plots.png`**: Test kümesi üzerinde 11 farklı modelin ve Ensemble (Topluluk) modelinin tahminlerini deneysel gerçek değerlerle kıyaslayan, üzerinde $R^2$, RMSE ve MAE değerleri yazılı **12 panelli regresyon grafik paneli**.
3. **`uncertainty_vs_error.png`**: Modeller arası belirsizlik (standart sapma) ile gerçek mutlak tahmin hatası arasındaki ilişkiyi gösteren ve Pearson korelasyonunu ($r$) hesaplayan saçılım grafiği.

---

## 📈 Google Sheets Entegrasyonu

Web sitesini canlıya aldıktan sonra 3. şahısların (arkadaşlarınızın, ziyaretçilerin) yaptığı tahmin verilerini kendi Google Sheets belgenizde toplamak isterseniz şu adımları izleyin:

1. **Google Form Oluşturun:** Google Drive'ınızda yeni bir Google Form oluşturun ve sırasıyla gerekli soruları ekleyin.
2. **Formu Yanıt Tablosuna Bağlayın:** Formun "Yanıtlar" sekmesinden e-tablo (Google Sheets) simgesine tıklayarak verilerin yazılacağı e-tabloyu oluşturun.
3. **Form URL'sini Alın:** Formun gönderim adresini edinin (Örn: `https://docs.google.com/forms/d/e/1FAIpQLSfXXXXXX/formResponse`). Bu adresi `app.py` dosyasında yer alan `GOOGLE_FORM_URL` değişkenine yapıştırın.
4. **Soru entry ID'lerini Eşleştirin:** Formun önizleme sayfasındaki HTML kodlarında yer alan `name="entry.XXXXXXXXXX"` parametrelerini bularak `app.py` içindeki `form_data` anahtarlarıyla eşleştirin.

---

## 📁 Proje Dosya Yapısı

```text
biooil_predictor/
├── train_model.py          # 11 makine öğrenmesi modelini eğitir
├── predict.py              # Komut satırı tahmin ve optimizasyon aracı (11 model)
├── app.py                  # İnteraktif Web Dashboard arayüzü (11 model destekli)
├── generate_plots.py       # Akademik grafikleri ve istatistikleri üretir
├── merged_dataset.csv      # Biyokütle eğitim veri kümesi
├── test_samples.csv        # Toplu tahmin için örnek girdi CSV şablonu
├── predictions_history.csv # İnteraktif tahminlerin geçmiş günlüğü (Otomatik oluşturulur)
├── requirements.txt        # Gerekli Python kütüphaneleri listesi
├── .gitignore              # Git tarafından takip edilmeyecek dosyalar listesi
├── .streamlit/
│   └── config.toml         # Streamlit sunucu ve arayüz yapılandırması
└── models/                 # train_model.py tarafından kaydedilen modeller ve metadata
    ├── linear_regression.pkl
    ├── ridge.pkl
    ├── knn.pkl
    ├── svr.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── extra_trees.pkl
    ├── gradient_boosting.pkl
    ├── xgboost.pkl
    ├── lightgbm.pkl
    ├── catboost.pkl
    └── metadata.json
```
