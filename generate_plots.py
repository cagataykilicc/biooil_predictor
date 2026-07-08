import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Configuration and constants
FEATURES = ["Cellulose_pct", "Hemicellulose_pct", "Lignin_pct", "ParticleSize_mm", "PyrolysisTemp_C"]
TARGET = "BioOilYield_pct"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODELS_DIR = Path("models")
DATA_PATH = Path("merged_dataset.csv")

MODEL_KEYS = [
    "linear_regression",
    "ridge",
    "knn",
    "svr",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]

# Set aesthetic styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["figure.dpi"] = 150

def load_data_and_models():
    """Loads the dataset and models, performs split, and returns test variables."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    
    # Train-test split (exact same split as train_model.py)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Load trained models
    models = {}
    for name in MODEL_KEYS:
        models[name] = joblib.load(MODELS_DIR / f"{name}.pkl")
        
    return df, X_test, y_test, models

def plot_correlation_matrix(df):
    """Generates a correlation matrix heatmap of features and target."""
    plt.figure(figsize=(8, 6.5))
    
    # Select columns to correlate
    cols_to_correlate = FEATURES + [TARGET]
    corr_df = df[cols_to_correlate].copy()
    
    # Translate column names to Turkish for nicer plots
    turkish_names = {
        "Cellulose_pct": "Selüloz (%)",
        "Hemicellulose_pct": "Hemiselüloz (%)",
        "Lignin_pct": "Lignin (%)",
        "ParticleSize_mm": "Parçacık Boyutu (mm)",
        "PyrolysisTemp_C": "Sıcaklık (°C)",
        "BioOilYield_pct": "Biyo-Yağ Verimi (%)"
    }
    corr_df = corr_df.rename(columns=turkish_names)
    
    # Compute correlation matrix
    corr_matrix = corr_df.corr(method="pearson")
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Korelasyon Katsayısı (r)"}
    )
    
    plt.title("Biyokütle Bileşenleri, İşletme Parametreleri ve Biyo-Yağ Verimi Korelasyon Matrisi", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    
    output_path = "correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Korelasyon matrisi kaydedildi: {output_path}")

def plot_regression_plots(X_test, y_test, models):
    """Generates scatter plots comparing Experimental vs. Predicted Yields for 11 models + ensemble (12 panels)."""
    # Compute predictions
    preds_list = []
    all_preds = {}
    for name in MODEL_KEYS:
        pred = models[name].predict(X_test)
        preds_list.append(pred)
        nice_name = name.replace('_', ' ').title()
        all_preds[nice_name] = pred
        
    ensemble_pred = np.mean(preds_list, axis=0)
    all_preds["Ensemble (Topluluk)"] = ensemble_pred
    
    # 3 rows, 4 columns = 12 subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    axes = axes.ravel()
    
    # Color palette for 12 subplots
    colors = sns.color_palette("tab20", 12)
    
    for i, (name, pred) in enumerate(all_preds.items()):
        ax = axes[i]
        
        # Calculate metrics
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        
        # Scatter plot
        ax.scatter(y_test, pred, color=colors[i], alpha=0.6, edgecolors="k", s=35, label="Gözlemler")
        
        # Identity line (y = x)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),  # min of both axes
            max(ax.get_xlim()[1], ax.get_ylim()[1]),  # max of both axes
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, zorder=2, linewidth=1.5, label="y = x (Mükemmel Uyum)")
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # Layout details
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Deneysel Biyo-Yağ Verimi (%)", fontsize=9)
        ax.set_ylabel("Tahmin Edilen Biyo-Yağ Verimi (%)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Text box for metrics
        metric_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} %\nMAE = {mae:.2f} %"
        ax.text(
            0.05, 0.95, metric_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray")
        )
        
        ax.legend(loc="lower right", fontsize=8)
        
    plt.suptitle("Regresyon Grafik Karşılaştırmaları (Deneysel vs. Tahmin Edilen Biyo-Yağ Verimi)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    output_path = "regression_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 12 Panelli Regresyon grafikleri karşılaştırması kaydedildi: {output_path}")

def plot_uncertainty_vs_error(X_test, y_test, models):
    """Plots Ensemble Uncertainty (Std Dev of 11 models) vs. Absolute Error."""
    preds_list = []
    for name in MODEL_KEYS:
        preds_list.append(models[name].predict(X_test))
        
    ensemble_pred = np.mean(preds_list, axis=0)
    
    # Calculate error and uncertainty
    abs_error = np.abs(y_test.values - ensemble_pred)
    uncertainty = np.std(preds_list, axis=0)
    
    # Pearson Correlation Coefficient (r) and p-value
    r_val, p_val = pearsonr(uncertainty, abs_error)
    
    plt.figure(figsize=(7.5, 6))
    
    # Scatter plot
    sns.scatterplot(x=uncertainty, y=abs_error, color="#673ab7", alpha=0.6, edgecolors="k", s=50, label="Gözlemler")
    
    # Fit regression line (Uncertainty vs Error)
    sns.regplot(
        x=uncertainty, y=abs_error, 
        scatter=False, color="#e91e63", 
        line_kws={"linewidth": 2, "label": "Eğilim Doğrusu"}
    )
    
    # Annotate stats
    plt.title("Ensemble Modeller Arası Belirsizlik (Standart Sapma) vs. Mutlak Tahmin Hatası", fontsize=11, fontweight="bold", pad=15)
    plt.xlabel("Modeller Arası Belirsizlik (Standart Sapma, %)", fontsize=10)
    plt.ylabel("Gerçek Mutlak Tahmin Hatası (%)", fontsize=10)
    
    # Text box for correlation metrics
    stats_text = f"Pearson Korelasyonu:\n$r$ = {r_val:.3f}\n$p$-değeri = {p_val:.5f}\n(p < 0.01: İstatistiksel Olarak Anlamlı)"
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
    )
    
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    output_path = "uncertainty_vs_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Belirsizlik vs. Hata grafiği kaydedildi: {output_path}")
    print(f"     Pearson r: {r_val:.4f}, p-value: {p_val:.6f}")

if __name__ == "__main__":
    print("--- Grafikler ve İstatistikler Oluşturuluyor ---")
    df, X_test, y_test, models = load_data_and_models()
    
    plot_correlation_matrix(df)
    plot_regression_plots(X_test, y_test, models)
    plot_uncertainty_vs_error(X_test, y_test, models)
    print("--- Tüm grafikler başarıyla oluşturuldu! ---")
