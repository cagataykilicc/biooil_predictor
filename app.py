import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Tarımsal Biyokütle Pirolizi — Bio-Yağ Verimi Tahmincisi",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = Path(__file__).parent / "models"

# Style customization (Premium CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1e3d59;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)


def save_to_history(sample_name, values, cb_pred, lgb_pred, xgb_pred, mean_pred, std_pred, min_pred, max_pred, opt_temp, opt_yield):
    """Web arayüzünde yapılan tahmini hem yerel CSV'ye hem de (ayarlanmışsa) Google Form aracılığıyla Google E-Tabloya kaydeder."""
    from datetime import datetime
    import requests
    history_file = Path("predictions_history.csv")
    
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Biomass_name": sample_name,
        "Cellulose_pct": values["Cellulose_pct"],
        "Hemicellulose_pct": values["Hemicellulose_pct"],
        "Lignin_pct": values["Lignin_pct"],
        "ParticleSize_mm": values["ParticleSize_mm"],
        "PyrolysisTemp_C": values["PyrolysisTemp_C"],
        "Predicted_BioOilYield_pct": mean_pred,
        "Prediction_Std": std_pred,
        "Prediction_Min": min_pred,
        "Prediction_Max": max_pred,
        "CatBoost_Pred": cb_pred,
        "LightGBM_Pred": lgb_pred,
        "XGBoost_Pred": xgb_pred,
        "Optimum_PyrolysisTemp_C": opt_temp,
        "Max_Predicted_BioOilYield_pct": opt_yield
    }
    
    df_new = pd.DataFrame([new_row])
    local_saved = False
    
    try:
        if history_file.exists():
            df_hist = pd.read_csv(history_file)
            df_combined = pd.concat([df_hist, df_new], ignore_index=True)
            df_combined.to_csv(history_file, index=False)
        else:
            df_new.to_csv(history_file, index=False)
        local_saved = True
    except Exception as e:
        st.error(f"Yerel dosyaya yazılırken hata oluştu: {str(e)}")
        
    # Google E-Tablo Entegrasyonu (Google Form POST İsteği)
    # Kendi Google Formunuzu oluşturduktan sonra aşağıdaki bilgileri güncelleyin:
    GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScVNCZuP38OYVdtaESONN2xnL7YE1ZMB0hh2p97-KPo_-5B_g/formResponse" # Örn: "https://docs.google.com/forms/d/e/1FAIpQLSfXXXXXX/formResponse"
    
    google_saved = False
    if GOOGLE_FORM_URL and "SİZİN_GOOGLE_FORM_URL" not in GOOGLE_FORM_URL:
        form_data = {
            "entry.1776230691": sample_name,
            "entry.334331509": round(float(values["Cellulose_pct"]), 2),
            "entry.644311460": round(float(values["Hemicellulose_pct"]), 2),
            "entry.410969399": round(float(values["Lignin_pct"]), 2),
            "entry.732774106": round(float(values["ParticleSize_mm"]), 2),
            "entry.1019577098": int(values["PyrolysisTemp_C"]),
            "entry.200322097": round(float(mean_pred), 2),
            "entry.36521110": round(float(std_pred), 2),
            "entry.1120845643": int(opt_temp),
            "entry.228553725": round(float(opt_yield), 2)
        }
        try:
            response = requests.post(GOOGLE_FORM_URL, data=form_data)
            if response.status_code == 200:
                google_saved = True
            else:
                st.warning(f"Google E-Tablo kaydı başarısız oldu (Durum Kodu: {response.status_code})")
        except Exception as e:
            st.warning(f"Google E-Tablo bağlantı hatası: {str(e)}")
            
    if local_saved:
        if google_saved:
            st.toast("💾 Tahmin hem Yerel CSV'ye hem de Google E-Tabloya kaydedildi!", icon="🚀")
        else:
            st.toast("💾 Tahmin yerel geçmiş dosyasına kaydedildi!", icon="✔️")


@st.cache_resource
def load_models_and_metadata():
    """Loads models and metadata and caches them for performance."""
    try:
        cb = joblib.load(MODELS_DIR / "catboost.pkl")
        lgb = joblib.load(MODELS_DIR / "lightgbm.pkl")
        xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
        with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {"catboost": cb, "lightgbm": lgb, "xgboost": xgb}, metadata
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {str(e)}")
        st.info("Lütfen önce model eğitimini tamamlayın: python train_model.py --data merged_dataset.csv")
        return None, None


models, metadata = load_models_and_metadata()

if models and metadata:
    st.title("🌱 Tarımsal Biyokütle Pirolizi — Bio-Yağ Verimi Tahmincisi")
    st.markdown(
        "Bu interaktif panel, tarımsal biyokütle bileşenlerini ve piroliz koşullarını "
        "kullanarak elde edilecek **bio-yağ verimini (%)** üç farklı gradient boosting modelinin "
        "(CatBoost, LightGBM, XGBoost) topluluk tahminiyle hesaplar."
    )

    # Setup layout
    col_input, col_results = st.columns([1, 2], gap="large")

    # Inputs (Left Sidebar or Left Column)
    with col_input:
        st.header("📋 Girdi Parametreleri")
        
        # Hazır biyokütle kütüphanesi
        FEEDSTOCK_PRESETS = {
            "Pirinç Sapı (Rice straw)": {"Cellulose_pct": 36.5, "Hemicellulose_pct": 24.0, "Lignin_pct": 15.4},
            "Buğday Sapı (Wheat straw)": {"Cellulose_pct": 38.6, "Hemicellulose_pct": 28.5, "Lignin_pct": 15.0},
            "Mısır Koçanı (Corn cob)": {"Cellulose_pct": 35.0, "Hemicellulose_pct": 28.0, "Lignin_pct": 15.0},
            "Fındık Kabuğu (Hazelnut shell)": {"Cellulose_pct": 26.8, "Hemicellulose_pct": 30.4, "Lignin_pct": 42.9},
            "Pamuk Sapı (Cotton stalk)": {"Cellulose_pct": 45.0, "Hemicellulose_pct": 21.0, "Lignin_pct": 30.0},
            "Ayçiçek Sapı (Sunflower stalk)": {"Cellulose_pct": 42.1, "Hemicellulose_pct": 29.7, "Lignin_pct": 13.4},
        }
        
        preset_choice = st.selectbox(
            "Hazır Biyokütle Kütüphanesi",
            ["Manuel Giriş"] + list(FEEDSTOCK_PRESETS.keys())
        )
        
        default_name = "Çam Talaşı"
        default_cel = 35.0
        default_hem = 25.0
        default_lig = 24.0
        
        if preset_choice in FEEDSTOCK_PRESETS:
            default_name = preset_choice
            default_cel = FEEDSTOCK_PRESETS[preset_choice]["Cellulose_pct"]
            default_hem = FEEDSTOCK_PRESETS[preset_choice]["Hemicellulose_pct"]
            default_lig = FEEDSTOCK_PRESETS[preset_choice]["Lignin_pct"]
            
        sample_name = st.text_input("Biyokütle / Numune Adı", default_name)
        
        st.subheader("Biyokütle Yapısı (%)")
        cel = st.slider("Cellulose (Selüloz) İçeriği (%)", 5.0, 70.0, default_cel, 0.1)
        hem = st.slider("Hemicellulose (Hemiselüloz) İçeriği (%)", 1.0, 60.0, default_hem, 0.1)
        lig = st.slider("Lignin İçeriği (%)", 1.0, 60.0, default_lig, 0.1)
        
        total_biomass = cel + hem + lig
        
        # Display sum feedback
        if total_biomass > 100.0:
            st.error(f"⚠️ Cel + Hem + Lig Toplamı: **%{total_biomass:.1f}** (>100). Fiziksel olarak imkansız!")
        elif total_biomass < 50.0:
            st.warning(f"⚠️ Cel + Hem + Lig Toplamı: **%{total_biomass:.1f}** (Çok düşük, tarımsal biyokütle için normalde %70-95 beklenir).")
        else:
            st.success(f"✔️ Cel + Hem + Lig Toplamı: **%{total_biomass:.1f}** (Geçerli aralıkta).")
            
        st.subheader("Piroliz Koşulları")
        ps = st.slider("Partikül Boyutu (mm)", 0.01, 12.00, 1.00, 0.01)
        pt = st.slider("Piroliz Sıcaklığı (°C)", 200, 1200, 500, 5)

        values = {
            "Cellulose_pct": cel,
            "Hemicellulose_pct": hem,
            "Lignin_pct": lig,
            "ParticleSize_mm": ps,
            "PyrolysisTemp_C": pt
        }

    # Results & Charts (Right Column)
    with col_results:
        # Perform Input Validation & Warnings
        st.header("⚡ Canlı Değerlendirme & Tahmin")
        st.markdown(f"**Analiz Edilen Numune:** `{sample_name}`")
        
        # Warnings container
        warnings = []
        stats = metadata["feature_stats"]
        
        # Check boundary/extrapolation
        for col in metadata["features"]:
            v = values[col]
            s = stats[col]
            if v < s["min"] or v > s["max"]:
                warnings.append(
                    f"⚠️ **{col}** = {v} -> Eğitim verisi sınırlarının **dışında** [{s['min']:.2f}, {s['max']:.2f}]. Tahmin güvenilirliği düşebilir."
                )
            elif v < s["p05"] or v > s["p95"]:
                warnings.append(
                    f"ℹ️ **{col}** = {v} -> Eğitim verisinin uç/seyrek bölgesinde (p05={s['p05']:.2f}, p95={s['p95']:.2f}). Tahmin kalitesi daha düşüktür."
                )

        if warnings:
            with st.expander("🔍 Girdi Durumu ve Güvenilirlik Uyarıları", expanded=True):
                for w in warnings:
                    st.markdown(w)
                    
        # Make predictions
        input_df = pd.DataFrame([values])
        cb_pred = float(models["catboost"].predict(input_df)[0])
        lgb_pred = float(models["lightgbm"].predict(input_df)[0])
        xgb_pred = float(models["xgboost"].predict(input_df)[0])
        
        preds = np.array([cb_pred, lgb_pred, xgb_pred])
        mean_pred = float(preds.mean())
        std_pred = float(preds.std())
        min_pred = float(preds.min())
        max_pred = float(preds.max())
        
        # Calculate temperature profile curve first so we have the optimum data
        temp_range = np.arange(250, 951, 10)
        curve_data = []
        for t_val in temp_range:
            test_val = values.copy()
            test_val["PyrolysisTemp_C"] = int(t_val)
            curve_data.append(test_val)
            
        curve_df = pd.DataFrame(curve_data)
        cb_curve = models["catboost"].predict(curve_df)
        lgb_curve = models["lightgbm"].predict(curve_df)
        xgb_curve = models["xgboost"].predict(curve_df)
        ensemble_curve = (cb_curve + lgb_curve + xgb_curve) / 3.0
        
        max_idx = np.argmax(ensemble_curve)
        opt_temp = int(temp_range[max_idx])
        opt_yield = float(ensemble_curve[max_idx])
        yield_diff = opt_yield - mean_pred
        
        # Main Metrics Row
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Ensemble Tahmini (Ortalama)</div>'
                f'<div class="metric-value">%{mean_pred:.2f}</div>'
                f'<div class="metric-label">Belirsizlik: ± {std_pred:.2f}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Beklenen Aralık</div>'
                f'<div class="metric-value">%{min_pred:.2f} - %{max_pred:.2f}</div>'
                f'<div class="metric-label">Modeller Arası Sapma</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
        st.write("")
        
        # Individual Models Row
        st.subheader("🤖 Bireysel Model Tahminleri")
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        with ind_col1:
            st.metric(label="CatBoost Tahmini", value=f"%{cb_pred:.2f}")
        with ind_col2:
            st.metric(label="LightGBM Tahmini", value=f"%{lgb_pred:.2f}")
        with ind_col3:
            st.metric(label="XGBoost Tahmini", value=f"%{xgb_pred:.2f}")

        # Tahmin Geçmişine Kaydetme Butonu
        st.write("")
        if st.button("💾 Tahmini Geçmiş Dosyasına Kaydet (predictions_history.csv)", use_container_width=True):
            save_to_history(sample_name, values, cb_pred, lgb_pred, xgb_pred, mean_pred, std_pred, min_pred, max_pred, opt_temp, opt_yield)

        # Interactive Chart: Temp vs Yield Profile
        st.subheader("📈 Sıcaklığa Bağlı Bio-Yağ Verim Profili")
        st.markdown(
            "Aşağıdaki grafik, girdiğiniz biyokütle özelliklerini sabit tutarak, "
            "piroliz sıcaklığının bio-yağ verimine etkisini gösterir. Kırmızı nokta mevcut sıcaklığı belirtmektedir."
        )
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(temp_range, ensemble_curve, label="Ensemble (Ortalama)", color="#1e3d59", linewidth=2.5)
        ax.fill_between(
            temp_range, 
            np.minimum(cb_curve, np.minimum(lgb_curve, xgb_curve)), 
            np.maximum(cb_curve, np.maximum(lgb_curve, xgb_curve)), 
            color="#1e3d59", 
            alpha=0.15, 
            label="Model Dağılım Aralığı"
        )
        
        # Highlight current point
        current_y = (cb_pred + lgb_pred + xgb_pred) / 3.0
        ax.scatter(pt, current_y, color="#ff6e40", s=100, zorder=5, edgecolors='black', label=f"Mevcut Nokta ({pt}°C, %{current_y:.1f})")
        
        ax.set_title("Piroliz Sıcaklığı vs. Bio-Yağ Verimi (%)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Piroliz Sıcaklığı (°C)", fontsize=10)
        ax.set_ylabel("Tahmin Edilen Bio-Yağ Verimi (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        
        st.pyplot(fig)
        
        st.subheader("🎯 Sıcaklık Optimizasyon Tavsiyesi")
        if abs(opt_temp - pt) <= 5:
            st.info(f"✔️ **Harika!** Seçtiğiniz sıcaklık ({pt}°C) zaten bu biyokütle için en ideal verimi veren optimum sıcaklığa (%{opt_yield:.2f} verim sağlayan {opt_temp}°C'ye) çok yakındır.")
        else:
            st.info(
                f"💡 **Tavsiye:** Bu biyokütle yapısı ve partikül boyutu ({ps} mm) için en yüksek bio-yağ verimini sağlayan "
                f"optimum sıcaklık **{opt_temp}°C**'dir (tahmini verim: **%{opt_yield:.2f}**).\n\n"
                f"Sıcaklığı mevcut `{pt}°C` değerinden `{opt_temp}°C` değerine ayarlayarak "
                f"verimi yaklaşık **%{abs(yield_diff):.2f}** oranında değiştirebilirsiniz."
            )
        
        # Metadata / Info Tab
        with st.expander("ℹ️ Model Performansı ve Eğitim Hakkında Bilgi", expanded=False):
            t_stats = metadata["target_stats"]
            st.markdown(
                f"**Veri Kümesi:** Model, Zhao (2024) ve Ortiz (2021) çalışmalarından derlenmiş "
                f"toplam **{metadata['n_train'] + metadata['n_test']} satır** veri üzerinde eğitilmiştir.\n\n"
                f"**Eğitim Verisi Gözlemlenen Aralık:** %{t_stats['min']:.1f} – %{t_stats['max']:.1f} "
                f"(Ortalama: %{t_stats['mean']:.1f})\n\n"
                f"**Test Seti R² Performans Skorları:**\n"
                f"- CatBoost: `{metadata['models']['catboost']['r2_test']:.3f}`\n"
                f"- LightGBM: `{metadata['models']['lightgbm']['r2_test']:.3f}`\n"
                f"- XGBoost: `{metadata['models']['xgboost']['r2_test']:.3f}`"
            )
else:
    st.warning("Uygulama çalıştırılamıyor. Lütfen modellerin eğitildiğinden emin olun.")
