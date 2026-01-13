"""
Detailed evaluation script - analiza wydajności modelu.
Oblicza metryki, tworzy wykresy, analizuje błędy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TabNetStrokeModel
from data_preparation import StrokeDataPreparator
from utils import print_classification_report, plot_confusion_matrix, plot_roc_curve


def load_preprocessing_info(filepath: str) -> dict:
    """
    Wczytuje informacje o preprocessingu.
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        Słownik z preprocessingiem
    """
    with open(filepath, 'rb') as f:
        prep_info = pickle.load(f)
    return prep_info


def load_feature_names(filepath: str) -> list:
    """
    Wczytuje nazwy cech.
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        Lista nazw cech
    """
    with open(filepath, 'rb') as f:
        feature_names = pickle.load(f)
    return feature_names


def prepare_test_data(
    csv_path: str,
    prep_info: dict,
    feature_names: list
) -> tuple:
    """
    Przygotowuje dane testowe korzystając z preprocessingu.
    
    Args:
        csv_path: Ścieżka do oryginalnego CSV
        prep_info: Informacje o preprocessingu
        feature_names: Nazwy cech
        
    Returns:
        Tuple (X_test, y_test, X_test_original)
    """
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Usuń 'id' jeśli istnieje
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Zachowaj oryginalną wersję (przed preprocessingiem)
    df_original = df.copy()
    
    # Obsłuż BMI
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    # Usuń brakujące wartości
    df = df.dropna()
    df_original = df_original.loc[df.index]
    
    # Zakoduj kategoryczni
    label_encoders = prep_info['label_encoders']
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    
    # Standaryzuj
    scaler = prep_info['scaler']
    numerical_features = prep_info['numerical_features']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    # Separuj cechy i target
    X = df[feature_names].values
    y = df['stroke'].values if 'stroke' in df.columns else None
    X_original = df_original[feature_names].values
    
    return X, y, X_original, df_original


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Oblicza wszystkie możliwe metryki klasyfikacji.
    
    📊 METRYKI (wyjaśnienie):
    ────────────────────────────────────────────────────
    • Accuracy: (TP + TN) / (TP + TN + FP + FN)
      - Procent poprawnych predykcji
      - ⚠️ Słaba gdy niezbalans klas!
    
    • Precision: TP / (TP + FP)
      - Gdy model powie "udار" - jak často ma rację?
      - Ważna gdy FP jest drogi (false alarms)
    
    • Recall (Sensitivity): TP / (TP + FN)
      - Ile rzeczywistych udarów model wykrywa?
      - Ważna w medycynie - nie chcemy przeoczyć!
    
    • F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
      - Harmonica średnia - równowaga między precision/recall
    
    • ROC-AUC: Area Under ROC Curve
      - Jak dobrze model rankuje (probability)
      - Niezależy od thresholdu (0.5)
      - Odporny na niezbalans klas
    
    • Specificity: TN / (TN + FP)
      - Ile rzeczywistych "bez udaru" wykrywa?
    ────────────────────────────────────────────────────
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje (0/1)
        y_pred_proba: Prawdopodobieństwa
        
    Returns:
        Słownik z wszystkimi metrykami
    """
    
    # Zmienne pomocnicze
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }
    
    return metrics


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, filepath: str = None) -> None:
    """
    Rysuje krzywą Precision-Recall.
    
    Przydatna dla niezbalansowanych zbiorów!
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred_proba: Prawdopodobieństwa
        filepath: Ścieżka do zapisania
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (lepsze dla niezbalansowanych danych)', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to: {filepath}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: dict, filepath: str = None) -> None:
    """
    Rysuje porównanie metryk.
    
    Args:
        metrics_dict: Słownik ze zbiorami (train, val, test)
        filepath: Ścieżka do zapisania
    """
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    data = []
    for dataset_name, metrics in metrics_dict.items():
        for metric in metrics_to_plot:
            data.append({
                'Dataset': dataset_name.upper(),
                'Metric': metric.upper(),
                'Value': metrics[metric]
            })
    
    df_metrics = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_metrics, x='Metric', y='Value', hue='Dataset', palette='Set2')
    plt.title('Porównanie metryk na zbiorach Train/Val/Test', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend(title='Dataset')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to: {filepath}")
    
    plt.show()


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    dataset_name: str = "Dataset"
) -> None:
    """
    Szczegółowa analiza predykcji.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        y_pred_proba: Prawdopodobieństwa
        dataset_name: Nazwa zbioru
    """
    
    print("\n" + "="*70)
    print(f"SZCZEGÓŁOWA ANALIZA - {dataset_name}")
    print("="*70)
    
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    print(f"\n📊 METRYKI PODSTAWOWE:")
    print(f"  • Accuracy:      {metrics['accuracy']:.4f}  (procent poprawnych)")
    print(f"  • Precision:     {metrics['precision']:.4f}  (gdy 'yes' - jak często ma rację)")
    print(f"  • Recall:        {metrics['recall']:.4f}  (ile rzeczywistych 'yes' wykryto)")
    print(f"  • F1-Score:      {metrics['f1']:.4f}  (harmonica średnia)")
    print(f"  • ROC-AUC:       {metrics['roc_auc']:.4f}  (niezależy od thresholdu)")
    print(f"  • Specificity:   {metrics['specificity']:.4f}  (ile rzeczywistych 'no' wykryto)")
    
    print(f"\n📈 MACIERZ POMYŁEK:")
    print(f"  • True Positives (TP):   {metrics['true_positive']:<5d} - Udary wykryte poprawnie")
    print(f"  • True Negatives (TN):   {metrics['true_negative']:<5d} - Brak udarów poprawnie")
    print(f"  • False Positives (FP):  {metrics['false_positive']:<5d} - Fałszywe alarmy")
    print(f"  • False Negatives (FN):  {metrics['false_negative']:<5d} - Przeoczyliśmy udary ⚠️")
    
    print(f"\n💡 INTERPRETACJA:")
    if metrics['false_negative'] > 0:
        print(f"  ⚠️  WAŻNE: Model przeoczy {metrics['false_negative']} udarów!")
        print(f"     To jest groźne - potrzebujemy wyższego recall'u!")
    
    if metrics['false_positive'] > 0:
        print(f"  ℹ️  Model zwróci {metrics['false_positive']} fałszywych alarmów")
        print(f"     To pochodzi, ale kontrolowalne (można zmienić threshold)")
    
    print("\n" + "="*70 + "\n")
    
    return metrics


def create_evaluation_report(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: TabNetStrokeModel,
    results_dir: str = "results"
) -> None:
    """
    Tworzy pełny raport ewaluacji.
    
    Args:
        X_test: Dane testowe
        y_test: Etykiety testowe
        model: Wytrenowany model
        results_dir: Katalog na wyniki
    """
    
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("TWORZENIE RAPORTU EWALUACJI")
    print("="*70 + "\n")
    
    # Predykcje
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metryki
    metrics = analyze_predictions(y_test, y_pred, y_pred_proba, "TEST SET")
    
    # Raport klasyfikacji
    print_classification_report(y_test, y_pred, "TEST SET")
    
    # Wizualizacje
    print("\n📊 Tworzenie wykresów...")
    
    plot_confusion_matrix(
        y_test, y_pred,
        filepath=str(results_dir / "eval_confusion_matrix.png")
    )
    
    plot_roc_curve(
        y_test, y_pred_proba,
        filepath=str(results_dir / "eval_roc_curve.png")
    )
    
    plot_precision_recall_curve(
        y_test, y_pred_proba,
        filepath=str(results_dir / "eval_precision_recall_curve.png")
    )
    
    # Feature importance
    feature_importance = model.get_feature_importance(X_test)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(15),
        x='importance',
        y='feature',
        palette='rocket'
    )
    plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(results_dir / "eval_feature_importance.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved to: {results_dir / 'eval_feature_importance.png'}")
    plt.show()
    
    # Zapisz feature importance do CSV
    feature_importance.to_csv(str(results_dir / "feature_importance.csv"), index=False)
    
    print(f"\n✓ Raport ewaluacji zakończony!")
    print(f"✓ Pliki zapisane w: {results_dir}")


def main():
    """
    Główna funkcja ewaluacji.
    """
    
    parser = argparse.ArgumentParser(description='Ewaluacja modelu stroke prediction')
    parser.add_argument('--csv-path', type=str, default='healthcare-dataset-stroke-data.csv',
                        help='Ścieżka do oryginalnego CSV')
    parser.add_argument('--model-path', type=str, default='models/tabnet_stroke_model.pkl',
                        help='Ścieżka do modelu')
    parser.add_argument('--prep-info-path', type=str, default='models/preprocessing_info.pkl',
                        help='Ścieżka do preprocessing info')
    parser.add_argument('--feature-names-path', type=str, default='models/feature_names.pkl',
                        help='Ścieżka do feature names')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Katalog na wyniki')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🏥 EWALUACJA MODELU PRZEWIDYWANIA UDARU")
    print("="*70 + "\n")
    
    # Wczytaj model
    if not Path(args.model_path).exists():
        print(f"❌ Błąd: Model nie znaleziony pod: {args.model_path}")
        return
    
    model = TabNetStrokeModel()
    model.load_model(args.model_path)
    print(f"✓ Model wczytany\n")
    
    # Wczytaj preprocessing info
    if not Path(args.prep_info_path).exists():
        print(f"❌ Błąd: Preprocessing info nie znaleziony: {args.prep_info_path}")
        return
    
    prep_info = load_preprocessing_info(args.prep_info_path)
    feature_names = load_feature_names(args.feature_names_path)
    print(f"✓ Preprocessing info wczytany\n")
    
    # Przygotuj dane testowe
    if not Path(args.csv_path).exists():
        print(f"❌ Błąd: CSV nie znaleziony: {args.csv_path}")
        return
    
    X_test, y_test, X_original, df_original = prepare_test_data(
        args.csv_path, prep_info, feature_names
    )
    print(f"✓ Dane testowe przygotowane: {X_test.shape[0]} próbek\n")
    
    # Ewaluacja
    create_evaluation_report(X_test, y_test, model, args.results_dir)


if __name__ == "__main__":
    main()