"""
Uruchamia cały pipeline: dane -> model -> trening -> ewaluacja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

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
    Główny pipeline treningu modelu TabNet do przewidywania ryzyka udaru mózgu w 8 krokach:.
    1. Przygotowanie katalogów
    2. Wczytanie i przygotowanie danych
    3. Budowanie modelu TabNet
    4. Trening modelu
    5. Ewaluacja na zbiorach train/val/test
    6. Wizualizacje
    7. Zapisanie modelu
    8. Podsumowanie wyników
    """
    print("przewidywanie ryzyka udaru mózgu - trening modelu TabNet:")
    print("")

    print("Krok 1: Przygotowanie katalogów")
    dirs = create_directories()
    print("Katalogi gotowe\n")

    print("Krok 2: Przygotowanie danych")
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
    
    # Sprawdź rozkład klas
    print("\rozkład klas w zbiorze treningowym:")
    check_class_imbalance(y_train.values)
    
    n_features = X_train.shape[1]
    print(f"\nPrzygotowanie danych zakończone")
    print(f"    Liczba cech: {n_features}")
    print(f"    Kształt train: {X_train.shape}")
    print(f"    Kształt val: {X_val.shape}")
    print(f"    Kształt test: {X_test.shape}\n")

    print("Krok 3: Budowanie modelu TabNet")
    
    model = TabNetStrokeModel(device_name="auto", seed=42)
    # Hiperparametry TabNet
    model.build_model(
        n_features=n_features,
        n_classes=2,                  # Dwie klasy: udar / brak udaru
        n_steps=4,                    # Liczba iteracyjnych kroków (main hyperparameter)
        n_independent=2,              # Niezależne neurony w każdym kroku
        n_shared=2,                   # Wspólne neurony między krokami
        width=128,                    # Szerokość warstw (liczba neuronów)
        virtual_batch_size=128,       
        momentum=0.02,                
        mask_type="sparsemax",
        lambda_sparse=1e-3            # Regularyzacja sparsity
    )
    print("Model zbudowany\n")

    print("Krok 4: Trening modelu(moze chwile potrwac)")
    
    history = model.train(
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        max_epochs=200,
        batch_size=32,
        learning_rate=2e-2,
        patience=20,                  #jeśli walidacja nie poprawia się przez 20 epok to stop
        verbose=True
    )
    
    print("Trening zakończony!\n")
    
    print("Krok 5: Ewaluacja modelu")
    
    # Train 
    y_train_pred = model.predict(X_train.values)
    y_train_proba = model.predict_proba(X_train.values)[:, 1]
    train_metrics = calculate_metrics(y_train.values, y_train_pred, y_train_proba)
    print_metrics(train_metrics, "TRAIN")
    
    # Val 
    y_val_pred = model.predict(X_val.values)
    y_val_proba = model.predict_proba(X_val.values)[:, 1]
    val_metrics = calculate_metrics(y_val.values, y_val_pred, y_val_proba)
    print_metrics(val_metrics, "VAL")
    
    # Test 
    y_test_pred = model.predict(X_test.values)
    y_test_proba = model.predict_proba(X_test.values)[:, 1]
    test_metrics = calculate_metrics(y_test.values, y_test_pred, y_test_proba)
    print_metrics(test_metrics, "TEST")
    
    # Zapisywanie metryk
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    save_metrics(all_metrics, str(dirs["results"] / "metrics.json"))
    
    print("Ewaluacja zakończona\n")
    
    print("Krok 6: Tworzenie wizualizacji")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test.values,
        y_test_pred,
        filepath=str(dirs["results"] / "confusion_matrix_test.png")
    )
    # ROC curve
    plot_roc_curve(
        y_test.values,
        y_test_proba,
        filepath=str(dirs["results"] / "roc_curve_test.png")
    )
    
    # Feature importance
    feature_importance = model.get_feature_importance(X_test)
    print("\n")
    print("ważność cech:")
    print(feature_importance.to_string(index=False))
    print()
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
    print(f"wykres waznosci cech zapisany do: {dirs['results'] / 'feature_importance.png'}")
    plt.show()
    
    print_classification_report(y_test.values, y_test_pred, "TEST SET")
    
    print("Wizualizacje gotowe\n")
    

    print("Krok 7: Zapisywanie modelu")
    
    model.save_model(str(dirs["models"] / "tabnet_stroke_model.pkl"))
    
    with open(str(dirs["models"] / "feature_names.pkl"), 'wb') as f:
        pickle.dump(preparator.feature_names, f)
    
    print("Model zapisany\n")
    

    print("\n" + "krok 8: Podsumowanie wyników treningu")
    print("trening zakończony pomyślnie!")
    print("")
    print(f"\nWyniki na zbiorze TEST:")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall: {test_metrics['recall']:.4f}")
    print(f"    F1-Score: {test_metrics['f1']:.4f}")
    print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    print(f"\n Pliki zapisane do:")
    print(f"    Model: {dirs['models'] / 'tabnet_stroke_model.pkl'}")
    print(f"    Preprocessing: {dirs['models'] / 'preprocessing_info.pkl'}")
    print(f"    Metryki: {dirs['results'] / 'metrics.json'}")
    print(f"    Wykresy: {dirs['results']}")

    print("\nAby sprawdzić wyniki przewidywan modelu dla nowych danych, użyj polecenia:")
    print("python3 src/infer.py --input example_input.csv")
    


if __name__ == "__main__":
    main()