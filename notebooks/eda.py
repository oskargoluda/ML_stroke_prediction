"""
Exploratory Data Analysis (EDA) - eksploracja danych stroke dataset.
Analiza rozkładów, korelacji, anomalii itp.

Uwaga: Ten plik to pseudo-notebook - normalnie byłby .ipynb
ale jest w formacie .py aby łatwo go uruchomić.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ustawienia
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath: str) -> pd.DataFrame:
    """Wczytaj dane."""
    print(f"📂 Wczytywanie danych z: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Załadowanych {len(df)} wierszy i {len(df.columns)} kolumn\n")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Podstawowe informacje o danych."""
    print("\n" + "="*70)
    print("1️⃣  PODSTAWOWE INFORMACJE")
    print("="*70)
    
    print(f"\n📊 Kształt: {df.shape[0]} wierszy × {df.shape[1]} kolumn")
    
    print(f"\n📋 Typy danych:")
    print(df.dtypes)
    
    print(f"\n❓ Brakujące wartości:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✓ Brak brakujących wartości")
    
    print(f"\n🔢 Duplikaty: {df.duplicated().sum()}")


def statistical_summary(df: pd.DataFrame) -> None:
    """Statystyki opisowe."""
    print("\n" + "="*70)
    print("2️⃣  STATYSTYKI OPISOWE")
    print("="*70)
    
    print(df.describe())
    
    print("\n📊 Metryki dodatkowe:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'id':
            print(f"\n{col}:")
            print(f"  • Mean:   {df[col].mean():.2f}")
            print(f"  • Median: {df[col].median():.2f}")
            print(f"  • Std:    {df[col].std():.2f}")
            print(f"  • Min:    {df[col].min():.2f}")
            print(f"  • Max:    {df[col].max():.2f}")
            print(f"  • Range:  {df[col].max() - df[col].min():.2f}")


def class_distribution(df: pd.DataFrame) -> None:
    """Rozkład zmiennej docelowej (stroke)."""
    print("\n" + "="*70)
    print("3️⃣  ROZKŁAD ZMIENNEJ DOCELOWEJ (STROKE)")
    print("="*70)
    
    stroke_counts = df['stroke'].value_counts()
    stroke_pct = df['stroke'].value_counts(normalize=True) * 100
    
    print(f"\n📊 Liczby przypadków:")
    for label in [0, 1]:
        count = stroke_counts.get(label, 0)
        pct = stroke_pct.get(label, 0)
        label_text = "Brak udaru" if label == 0 else "UDAR ⚠️"
        print(f"  {label_text:.<20} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n⚠️  NIEZBALANS KLAS: {stroke_counts[0]/stroke_counts[1]:.1f}:1")
    print("   (19.5 razy więcej przypadków bez udaru)")
    
    # Wizualizacja
    plt.figure(figsize=(8, 5))
    colors = ['#2ecc71', '#e74c3c']
    stroke_counts.plot(kind='bar', color=colors)
    plt.title('Rozkład Zmiennej Docelowej (Stroke)', fontsize=14, fontweight='bold')
    plt.xlabel('Stroke (0=No, 1=Yes)')
    plt.ylabel('Liczba przypadków')
    plt.xticks([0, 1], ['Brak udaru', 'UDAR'], rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/eda_stroke_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Wykres zapisany do: results/eda_stroke_distribution.png")
    plt.show()


def age_distribution(df: pd.DataFrame) -> None:
    """Analiza wieku."""
    print("\n" + "="*70)
    print("4️⃣  ANALIZA WIEKU (AGE)")
    print("="*70)
    
    print(f"\nWiek:")
    print(f"  • Średnia: {df['age'].mean():.1f} lat")
    print(f"  • Mediana: {df['age'].median():.1f} lat")
    print(f"  • Zakres:  {df['age'].min():.0f} - {df['age'].max():.0f} lat")
    
    # Wizualizacja
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["age"].mean():.1f}')
    axes[0].set_title('Rozkład Wieku', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Wiek (lata)')
    axes[0].set_ylabel('Liczba osób')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Wiek a Stroke
    df.boxplot(column='age', by='stroke', ax=axes[1])
    axes[1].set_title('Wiek vs Udar', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Stroke (0=No, 1=Yes)')
    axes[1].set_ylabel('Wiek (lata)')
    
    plt.tight_layout()
    plt.savefig('results/eda_age_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Wykresy zapisane do: results/eda_age_analysis.png")
    plt.show()


def glucose_and_bmi(df: pd.DataFrame) -> None:
    """Analiza glukozzy i BMI."""
    print("\n" + "="*70)
    print("5️⃣  ANALIZA GLUKOZZY I BMI")
    print("="*70)
    
    # Glukoza
    print(f"\nGlukoza we krwi (avg_glucose_level):")
    print(f"  • Średnia: {df['avg_glucose_level'].mean():.1f} mg/dL")
    print(f"  • Mediana: {df['avg_glucose_level'].median():.1f} mg/dL")
    print(f"  • Zakres:  {df['avg_glucose_level'].min():.0f} - {df['avg_glucose_level'].max():.0f} mg/dL")
    
    # BMI
    print(f"\nBMI (Body Mass Index):")
    print(f"  • Średnia: {df['bmi'].mean():.1f}")
    print(f"  • Mediana: {df['bmi'].median():.1f}")
    print(f"  • Zakres:  {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    
    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram glukozzy
    axes[0, 0].hist(df['avg_glucose_level'], bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Rozkład Glukozzy we Krwi', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Glukoza (mg/dL)')
    axes[0, 0].set_ylabel('Liczba osób')
    axes[0, 0].grid(alpha=0.3)
    
    # Histogram BMI
    axes[0, 1].hist(df['bmi'].dropna(), bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Rozkład BMI', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Liczba osób')
    axes[0, 1].grid(alpha=0.3)
    
    # Glukoza vs Stroke
    df.boxplot(column='avg_glucose_level', by='stroke', ax=axes[1, 0])
    axes[1, 0].set_title('Glukoza vs Udar', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Stroke (0=No, 1=Yes)')
    axes[1, 0].set_ylabel('Glukoza (mg/dL)')
    
    # BMI vs Stroke
    df.boxplot(column='bmi', by='stroke', ax=axes[1, 1])
    axes[1, 1].set_title('BMI vs Udar', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Stroke (0=No, 1=Yes)')
    axes[1, 1].set_ylabel('BMI')
    
    plt.tight_layout()
    plt.savefig('results/eda_glucose_bmi.png', dpi=300, bbox_inches='tight')
    print("✓ Wykresy zapisane do: results/eda_glucose_bmi.png")
    plt.show()


def categorical_features(df: pd.DataFrame) -> None:
    """Analiza cech kategorycznych."""
    print("\n" + "="*70)
    print("6️⃣  CECHY KATEGORYCZNE")
    print("="*70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        print(df[col].value_counts())
        print(f"  Proporcja (%):")
        print((df[col].value_counts(normalize=True) * 100).round(2))
    
    # Wizualizacja
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    categorical_cols_list = list(categorical_cols)
    for idx, col in enumerate(categorical_cols_list):
        if idx < len(axes):
            df[col].value_counts().plot(kind='bar', ax=axes[idx], color='teal', alpha=0.7)
            axes[idx].set_title(f'Rozkład {col}', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Liczba')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
    
    # Usuń puste subploty
    for idx in range(len(categorical_cols_list), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('results/eda_categorical_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Wykresy zapisane do: results/eda_categorical_features.png")
    plt.show()


def binary_features(df: pd.DataFrame) -> None:
    """Analiza cech binarnych."""
    print("\n" + "="*70)
    print("7️⃣  CECHY BINARNE")
    print("="*70)
    
    binary_cols = ['hypertension', 'heart_disease', 'stroke']
    
    for col in binary_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            counts = df[col].value_counts()
            for label in [0, 1]:
                count = counts.get(label, 0)
                pct = (count / len(df)) * 100
                label_text = "No/False" if label == 0 else "Yes/True"
                print(f"  {label_text:.<20} {count:>5} ({pct:>5.1f}%)")
    
    # Wizualizacja
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, col in enumerate(binary_cols):
        colors = ['#2ecc71', '#e74c3c']
        df[col].value_counts().plot(kind='bar', ax=axes[idx], color=colors, alpha=0.7)
        axes[idx].set_title(f'{col.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Liczba')
        axes[idx].set_xticklabels(['No/False', 'Yes/True'], rotation=0)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/eda_binary_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Wykresy zapisane do: results/eda_binary_features.png")
    plt.show()


def correlation_analysis(df: pd.DataFrame) -> None:
    """Analiza korelacji."""
    print("\n" + "="*70)
    print("8️⃣  ANALIZA KORELACJI")
    print("="*70)
    
    # Wybierz tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Macierz korelacji
    corr_matrix = numeric_df.corr()
    
    # Korelacja z targetem (stroke)
    stroke_corr = corr_matrix['stroke'].sort_values(ascending=False)
    
    print(f"\nKorelacja cech z STROKE:")
    for feature, corr_val in stroke_corr.items():
        if feature != 'stroke':
            print(f"  {feature:.<30} {corr_val:>7.4f}")
    
    # Wizualizacja - heatmapa
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Macierz Korelacji Cech Numerycznych', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Heatmapa korelacji zapisana do: results/eda_correlation_matrix.png")
    plt.show()


def main():
    """Główna funkcja EDA."""
    
    print("\n" + "="*70)
    print("🏥 EKSPLORACYJNA ANALIZA DANYCH (EDA)")
    print("   Healthcare Dataset - Stroke Prediction")
    print("="*70 + "\n")
    
    # Utwórz katalog results
    Path('results').mkdir(exist_ok=True)
    
    # Wczytaj dane
    df = load_data('healthcare-dataset-stroke-data.csv')
    
    # Analizy
    basic_info(df)
    statistical_summary(df)
    class_distribution(df)
    age_distribution(df)
    glucose_and_bmi(df)
    categorical_features(df)
    binary_features(df)
    correlation_analysis(df)
    
    print("\n" + "="*70)
    print("✅ EDA ZAKOŃCZONA!")
    print("="*70)
    print("\n📊 Wykresy zapisane w katalogu: results/")
    print("   - eda_stroke_distribution.png")
    print("   - eda_age_analysis.png")
    print("   - eda_glucose_bmi.png")
    print("   - eda_categorical_features.png")
    print("   - eda_binary_features.png")
    print("   - eda_correlation_matrix.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()