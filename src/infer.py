"""
Skrypt predykcji ryzyka udaru z wykorzystaniem modelu TabNet dla nowych danych.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from model import TabNetStrokeModel
from data_preparation import StrokeDataPreparator


def load_preprocessing_info(filepath: str) -> dict:
    """
    Wczytuje informacje o preprocessingu (encodery, scaler).
    """
    with open(filepath, 'rb') as f:
        prep_info = pickle.load(f)
    
    return prep_info


def load_feature_names(filepath: str) -> list:
    """
    Wczytuje nazwy cech.
    """
    with open(filepath, 'rb') as f:
        feature_names = pickle.load(f)
    
    return feature_names


def predict_stroke_risk(
    model: TabNetStrokeModel,
    X: np.ndarray,
    threshold: float = 0.70
) -> pd.DataFrame:
    """
    Robi predykcje na przygotowanych danych.
    """
    
    print("")
    print("prdykcja ryzyka udaru")
    print("")

    y_proba = model.predict_proba(X)
    
    stroke_probability = y_proba[:, 1]

    stroke_prediction = (stroke_probability >= threshold).astype(int)
    
    #tworzenie DataFrame'u z wynikami
    results = pd.DataFrame({
        'stroke_probability': stroke_probability,
        'stroke_prediction': stroke_prediction,
        'risk_category': pd.cut(
            stroke_probability,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })
    
    print(f"Predykcje wykonane")
    print(f"\nStatystyki predykcji:")
    print(f"    Średnie prawdopodobieństwo udaru: {stroke_probability.mean():.4f}")
    print(f"    Min: {stroke_probability.min():.4f}")
    print(f"    Max: {stroke_probability.max():.4f}")
    print(f"    Std: {stroke_probability.std():.4f}")

    return results


def main():
    """
    Główna funkcja inferencji.
    """
    
    parser = argparse.ArgumentParser(
        description='Przewidywanie ryzyka udaru na nowych danych'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='example_input.csv',
        help='Ścieżka do pliku CSV z danymi wejściowymi'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Ścieżka do pliku CSV z predykcjami'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.70,
        help='Próg decyzyjny (domyślnie 0.70)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/tabnet_stroke_model.pkl',
        help='Ścieżka do wytrenowanego modelu'
    )
    parser.add_argument(
        '--prep-info-path',
        type=str,
        default='models/preprocessing_info.pkl',
        help='Ścieżka do informacji o preprocessingu'
    )
    parser.add_argument(
        '--feature-names-path',
        type=str,
        default='models/feature_names.pkl',
        help='Ścieżka do nazw cech'
    )
    
    args = parser.parse_args()
    

    print("Wczytywanie modelu")
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        if Path(str(model_path) + ".zip").exists():
            model_path = Path(str(model_path) + ".zip")
        else:
            print(f"Błąd: Model nie znaleziony pod: {args.model_path}")
            return
    
    model = TabNetStrokeModel()
    model.load_model(str(model_path))
    print("Model wczytany\n")

    print("Wczytywanie informacji o preprocessingu")
    
    if not Path(args.prep_info_path).exists():
        print(f"Błąd: Plik preprocessingu nie znaleziony: {args.prep_info_path}")
        return
    
    prep_info = load_preprocessing_info(args.prep_info_path)
    feature_names = load_feature_names(args.feature_names_path)
    print("Informacje wczytane\n")

    print("Wczytywanie danych wejściowych")
    
    if not Path(args.input).exists():
        print(f"Błąd: Plik danych nie znaleziony: {args.input}")
        return
    
    df_input = pd.read_csv(args.input)
    print(f"Wczytane dane: {df_input.shape[0]} próbek, {df_input.shape[1]} kolumn\n")
    
    print(f"Przygotowanie danych")
    preparator = StrokeDataPreparator()
    preparator.load_preprocessing_info(args.prep_info_path)
    preparator.feature_names = feature_names
    
    df_transformed = preparator.transform(df_input)
    X = df_transformed.values
    print(f"Dane przygotowane: {X.shape[0]} próbek, {X.shape[1]} cech\n")


    predictions = predict_stroke_risk(model, X, threshold=args.threshold)
    

    df_output = df_transformed.copy()
    
    df_output['stroke_probability'] = predictions['stroke_probability'].values
    df_output['stroke_prediction'] = predictions['stroke_prediction'].values
    df_output['risk_category'] = predictions['risk_category'].values
    

    df_output.to_csv(args.output, index=False)
    
    print("")
    print("predykcja zapisana")
    print("")
    print(f"\nWyniki:")
    print(df_output[['stroke_probability', 'stroke_prediction', 'risk_category']])
    print(f"\ndane zapisany w pliku: {args.output}")


if __name__ == "__main__":
    main()