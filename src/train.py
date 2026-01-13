"""
Main training script for stroke prediction model.
Uruchamia cały pipeline: dane → model → trening → ewaluacja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

# Dodaj src do path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preparation import StrokeDataPreparator
from model import TabNetStrokeModel, evaluate_model
from utils import (
    create_directories, calculate_metrics, print_metrics, 
    save_metrics, plot_confusion_matrix, plot_roc_curve,
    print_classification_report, check_class_imbalance
)


def main():
    """
    Główna funkcja - pipeline treningu.
    """
    
    print("\n" + "="*70)
    print("🏥 PRZEWIDYWANIE RYZYKA UDARU - TABNET")
    print("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 1: Przygotowanie katalogów
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 1: Przygotowanie katalogów...")
    dirs = create_directories()
    print("✓ Katalogi gotowe\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 2: Wczytanie i przygotowanie danych
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 2: Przygotowanie danych...")
    
    # Ścieżka do pliku danych
    data_path = "data/raw/healthcare-dataset-stroke-data.csv"
    
    if not Path(data_path).exists():
        print(f"Błąd: Plik {data_path} nie znaleziony!")
        print("Upewnij się, że plik znajduje się w bieżącym katalogu")
        return
    
    # Inicjalizuj preparator danych
    preparator = StrokeDataPreparator(random_state=42)
    
    # Pełny pipeline przygotowania
    X_train, X_val, X_test, y_train, y_val, y_test = preparator.prepare(data_path)
    
    # Zapisz informacje o preprocessingu (będą potrzebne do inferencji)
    preparator.save_preprocessing_info(str(dirs["models"] / "preprocessing_info.pkl"))
    
    # Sprawdź niezbalans klas
    print("\nNiezbalans klas w zbiorze treningowym:")
    check_class_imbalance(y_train.values)
    
    n_features = X_train.shape[1]
    print(f"\n✓ Przygotowanie danych zakończone")
    print(f"  • Liczba cech: {n_features}")
    print(f"  • Kształt train: {X_train.shape}")
    print(f"  • Kształt val: {X_val.shape}")
    print(f"  • Kształt test: {X_test.shape}\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 3: Budowanie modelu
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 3: Budowanie modelu TabNet...")
    
    model = TabNetStrokeModel(device_name="auto", seed=42)
    
    # Hiperparametry TabNet
    # Te parametry można tunować dla lepszych wyników
    model.build_model(
        n_features=n_features,
        n_classes=2,
        n_steps=5,                    # Liczba iteracyjnych kroków (main hyperparameter)
        n_independent=2,              # Niezależne neurony w każdym kroku
        n_shared=2,                   # Wspólne neurony między krokami
        width=128,                    # Szerokość warstw (liczba neuronów)
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        lambda_sparse=1e-3            # Regularyzacja sparsity
    )
    print("✓ Model zbudowany\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 4: Trening modelu
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 4: Trening modelu...")
    
    history = model.train(
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        max_epochs=200,
        batch_size=32,
        learning_rate=2e-2,
        patience=20,                  # Early stopping: jeśli walidacja nie poprawia się przez 20 epok
        verbose=True
    )
    
    print("✓ Trening zakończony\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 5: Ewaluacja na zbiorach train/val/test
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 5: Ewaluacja modelu...")
    
    # Train metrics
    y_train_pred = model.predict(X_train.values)
    y_train_proba = model.predict_proba(X_train.values)[:, 1]
    train_metrics = calculate_metrics(y_train.values, y_train_pred, y_train_proba)
    print_metrics(train_metrics, "TRAIN")
    
    # Val metrics
    y_val_pred = model.predict(X_val.values)
    y_val_proba = model.predict_proba(X_val.values)[:, 1]
    val_metrics = calculate_metrics(y_val.values, y_val_pred, y_val_proba)
    print_metrics(val_metrics, "VAL")
    
    # Test metrics
    y_test_pred = model.predict(X_test.values)
    y_test_proba = model.predict_proba(X_test.values)[:, 1]
    test_metrics = calculate_metrics(y_test.values, y_test_pred, y_test_proba)
    print_metrics(test_metrics, "TEST")
    
    # Zapisz metryki
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    save_metrics(all_metrics, str(dirs["results"] / "metrics.json"))
    
    print("✓ Ewaluacja zakończona\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 6: Wizualizacje
    # ════════════════════════════════════════════════════════════════════════
    print("Krok 6: Tworzenie wizualizacji...")
    
    # Confusion matrix - zbiór testowy
    plot_confusion_matrix(
        y_test.values,
        y_test_pred,
        filepath=str(dirs["results"] / "confusion_matrix_test.png")
    )
    
    # ROC curve - zbiór testowy
    plot_roc_curve(
        y_test.values,
        y_test_proba,
        filepath=str(dirs["results"] / "roc_curve_test.png")
    )
    
    # Feature importance
    feature_importance = model.get_feature_importance(X_test.values)
    print("\n" + "="*70)
    print("WAŻNOŚĆ CECH (Feature Importance)")
    print("="*70)
    print(feature_importance.to_string(index=False))
    print()
    
    # Wizualizuj feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(10),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title('Top 10 Most Important Features (TabNet)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(dirs["results"] / "feature_importance.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved to: {dirs['results'] / 'feature_importance.png'}")
    plt.show()
    
    # Classification report
    print_classification_report(y_test.values, y_test_pred, "TEST SET")
    
    print("✓ Wizualizacje gotowe\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # KROK 7: Zapisanie modelu
    # ════════════════════════════════════════════════════════════════════════
    print("💾 Krok 7: Zapisywanie modelu...")
    
    model.save_model(str(dirs["models"] / "tabnet_stroke_model.pkl"))
    
    # Zapisz feature names do later use
    with open(str(dirs["models"] / "feature_names.pkl"), 'wb') as f:
        pickle.dump(preparator.feature_names, f)
    
    print("✓ Model zapisany\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # PODSUMOWANIE
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TRENING ZAKOŃCZONY POMYŚLNIE!")
    print("="*70)
    print(f"\n📊 Wyniki na zbiorze TEST:")
    print(f"  • Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  • Precision: {test_metrics['precision']:.4f}")
    print(f"  • Recall: {test_metrics['recall']:.4f}")
    print(f"  • F1-Score: {test_metrics['f1']:.4f}")
    print(f"  • ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    print(f"\n📁 Pliki zapisane:")
    print(f"  • Model: {dirs['models'] / 'tabnet_stroke_model.pkl'}")
    print(f"  • Preprocessing: {dirs['models'] / 'preprocessing_info.pkl'}")
    print(f"  • Metryki: {dirs['results'] / 'metrics.json'}")
    print(f"  • Wykresy: {dirs['results']}")
    
    print(f"\n💡 Następne kroki:")
    print(f"  1. Sprawdź wykresy w katalogu {dirs['results']}")
    print(f"  2. Uruchom python3 src/infer.py --input example_input.csv")
    print(f"  3. Przejrzyj sprawozdanie w README.md")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()