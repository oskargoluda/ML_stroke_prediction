"""
Utility functions for stroke prediction project.
Zawiera: preprocessing, metrics, plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from pathlib import Path
import json


def create_directories(base_path: str = ".") -> dict:
    """
    Tworzy wymagane katalogi dla projektu.
    
    Args:
        base_path: Ścieżka bazowa projektu
        
    Returns:
        Dictionary ze ścieżkami katalogów
    """
    dirs = {
        "data_raw": Path(base_path) / "data" / "raw",
        "data_processed": Path(base_path) / "data" / "processed",
        "models": Path(base_path) / "models",
        "results": Path(base_path) / "results",
        "src": Path(base_path) / "src"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict:
    """
    Oblicza metryki klasyfikacji.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje (0 lub 1)
        y_pred_proba: Prawdopodobieństwa (opcjonalne, do ROC-AUC)
        
    Returns:
        Słownik z metrykami
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC wymaga prawdopodobieństw
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_metrics(metrics: dict, dataset_name: str = "Dataset") -> None:
    """
    Wypisuje metryki w czytelnym formacie.
    
    Args:
        metrics: Słownik metryk
        dataset_name: Nazwa zbioru danych
    """
    print(f"\n{'='*50}")
    print(f"Metryki dla: {dataset_name}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"{key.upper():.<30} {value:.4f}")
    print(f"{'='*50}\n")


def save_metrics(metrics: dict, filepath: str) -> None:
    """
    Zapisuje metryki do pliku JSON.
    
    Args:
        metrics: Słownik metryk
        filepath: Ścieżka do pliku
    """
    # Konwersja numpy types do Python types (aby JSON mogła je zapisać)
    metrics_serializable = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in metrics.items()
    }
    
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"✓ Metryki zapisane do: {filepath}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, filepath: str = None) -> None:
    """
    Rysuje macierz pomyłek.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        filepath: Opcjonalna ścieżka do zapisania
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {filepath}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, filepath: str = None) -> None:
    """
    Rysuje krzywą ROC.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred_proba: Prawdopodobieństwa
        filepath: Opcjonalna ścieżka do zapisania
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {filepath}")
    
    plt.show()


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "Dataset") -> None:
    """
    Wypisuje szczegółowy raport klasyfikacji.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        dataset_name: Nazwa zbioru danych
    """
    print(f"\n{'='*60}")
    print(f"Classification Report - {dataset_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['No Stroke', 'Stroke']))
    print(f"{'='*60}\n")


def check_class_imbalance(y: np.ndarray) -> None:
    """
    Sprawdza niezbalans klas.
    
    Args:
        y: Etykiety
    """
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nRozkład klas:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  Klasa {label}: {count:5d} próbek ({percentage:5.2f}%)")
    print()