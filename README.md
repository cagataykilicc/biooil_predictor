# 🌱 Tarımsal Biyokütle Pirolizi — Bio-Yağ Verimi Tahmin Programı

Tarımsal biyokütle pirolizinden elde edilecek **bio-yağ verimini (%)** tahmin eden ve piroliz sıcaklığını optimize eden interaktif bir yapay zeka aracıdır. 

Üç güçlü gradient boosting modelinin (CatBoost, LightGBM, XGBoost) topluluk (ensemble) tahminiyle çalışır ve modeller arası varyansı tahmin belirsizliği (sapma) olarak sunar.

---

## 🚀 Yeni Eklenen Özellikler

*   **🌐 İnteraktif Web Dashboard (Streamlit):** Girdileri sliders yardımıyla kontrol edebileceğiniz, anlık tahmin grafiklerini ve sıcaklık-verim analizlerini izleyebileceğiniz modern bir web uygulaması (`app.py`).
*   **🌾 Hazır Biyokütle Kütüphanesi (Presets):** Yaygın tarımsal biyokütlelerin (Pirinç Sapı, Buğday Sapı, Mısır Koçanı, Fındık Kabuğu vb.) kimyasal bileşen yüzdeleri sisteme tanımlanmıştır. Tek tıkla otomatik doldurulabilir.
*   **🎯 Sıcaklık Optimizasyon Motoru:** Biyokütle bileşimini sabit tutarak 300°C - 900°C arasındaki tüm sıcaklıkları simüle eder ve en yüksek verimi alabileceğiniz **optimum piroliz sıcaklığını** önerir.
*   **🏷️ Numune İsmi Tanımlama:** Tahminlerinizi isimlendirebilir ve raporları bu etiketle kaydedebilirsiniz.
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

## 📊 Örnek Çıktı Yapısı (Komut Satırı)

```text
======================================================================
  TAHMİN SONUCU
======================================================================
  Biyokütle Adı:  Fındık Kabuğu Denemesi
  Bio-yağ verimi: 43.63% +/- 0.35
  Aralık:         [43.32, 44.11] %

======================================================================
  SICAKLIK OPTİMİZASYON TAVSİYESİ
======================================================================
  Girdiğiniz bileşim ve partikül boyutu (0.5 mm) için:
    En yüksek verimi sağlayan optimum sıcaklık: 455°C
    Bu sıcaklıktaki tahmini bio-yağ verimi:    %47.58
    >> Tavsiye: Sıcaklığı 500°C değerinden 455°C değerine ayarlamak,
       verimi yaklaşık %3.95 oranında artırabilir.
```

---

## 📁 Proje Dosya Yapısı

```text
biooil_predictor/
├── train_model.py          # Makine öğrenmesi modellerini eğitir
├── predict.py              # Komut satırı tahmin ve optimizasyon aracı
├── app.py                  # İnteraktif Web Dashboard arayüzü (Streamlit)
├── merged_dataset.csv      # Biyokütle eğitim veri kümesi
├── test_samples.csv        # Toplu tahmin için örnek girdi CSV şablonu
├── predictions_history.csv # İnteraktif tahminlerin geçmiş günlüğü (Otomatik oluşturulur)
├── requirements.txt        # Gerekli Python kütüphaneleri listesi
├── .gitignore              # Git tarafından takip edilmeyecek dosyalar listesi
├── .streamlit/
│   └── config.toml         # Streamlit sunucu ve arayüz yapılandırması
└── models/                 # train_model.py tarafından kaydedilen modeller
    ├── catboost.pkl
    ├── lightgbm.pkl
    ├── xgboost.pkl
    └── metadata.json
```
