"""
Model definition and utilities for TabNet stroke prediction.
Zawiera: inicjalizacja TabNet, callback'i, funkcje pomocnicze.
"""

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import warnings
warnings.filterwarnings('ignore')


class TabNetStrokeModel:
    """
    Wrapper wokół pytorch_tabnet.TabNetClassifier dla przewidywania udarów.
    Zawiera:
    -Inicjalizację modelu z hiperparametrami
    -Trening z early stopping
    -Predykcje
    -Zapisywanie/wczytywanie modelu
    """
    
    def __init__(self, device_name: str = "auto", seed: int = 42):
        """
        Inicjalizuje model TabNet.
        """
        self.seed = seed
        self.device_name = device_name
        self.model = None
        self.history = None
        
        # Ustawiamy seed dla PyTorch
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def build_model(
        self,
        n_features: int,
        n_classes: int = 2,
        n_steps: int = 4,
        n_independent: int = 2,
        n_shared: int = 2,
        n_decision_steps: int = 5,
        width: int = 128,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "softmax",
        lambda_sparse: float = 1e-3
    ) -> TabNetClassifier:
        """
        Buduje model TabNet.
        
        🔬 MATEMATYKA TABNet:
        ─────────────────────────────────────────────────────────────
        TabNet to sekwencyjny model feature selection + decision steps.
        
        W każdym kroku i:
        1. Prior mask (zapamiętana historia):  prior_i
        2. Feature mask generation:  m_i = sparsemax(prior_i + decision_out_i)
        3. Sparse features:  f_i = m_i ⊙ X  (⊙ = element-wise mult)
        4. Decision step:  d_i = DecisionStep(f_i)
        5. Update prior:  prior_{i+1} = prior_i - ε * m_i
        
        Output = aggregation(d_1, d_2, ..., d_n_steps)
        ─────────────────────────────────────────────────────────────
        
        Args:
            n_features: Liczba cech wejściowych
            n_classes: Liczba klas (2 dla stroke/no-stroke)
            n_steps: Liczba decision steps (iteracyjny feature selection)
            n_independent: Neurony niezależne w każdym decision step
            n_shared: Neurony wspólne między decision steps
            n_decision_steps: Alias dla n_steps (niektóre wersje używają tego)
            width: Szerokość (liczba neuronów) w hidden layers
            virtual_batch_size: Batch size dla ghost batch normalization
            momentum: Momentum dla batch normalizacji
            mask_type: "softmax" lub "sparsemax" dla masks
            lambda_sparse: Regularyzacja rzadkości (sparsity penalty)
            
        Returns:
            Skonfigurowany model TabNetClassifier
        """
        self.virtual_batch_size = virtual_batch_size
        
        self.model = TabNetClassifier(
            n_d=width,                    # Width of decision step
            n_a=width,                    # Width of feature transformer
            n_steps=n_steps,              # Number of decision steps (główny hiperparametr!)
            n_independent=n_independent,  # Number of independent GLUs
            n_shared=n_shared,            # Number of shared GLUs
            momentum=momentum,
            mask_type=mask_type,
            lambda_sparse=lambda_sparse,  # Regularyzacja sparsity
            device_name=self.device_name,
            seed=self.seed,
            verbose=0,
            optimizer_params=dict(lr=2e-2)
        )
        
        print(f"Model TabNet stworzony")
        print(f"\nArchitektura:")
        print(f"    Decision steps: {n_steps}")
        print(f"    Width (neurony na warstwę): {width}")
        print(f"    Sparsity regularization (λ): {lambda_sparse}")
        print(f"    Independent GLUs: {n_independent}")
        print(f"    Shared GLUs: {n_shared}")
        print(f"    Mask type: {mask_type}")
        print(f"    Virtual batch size: {virtual_batch_size}")
        print(f"    Input features: {n_features}")
        print(f"    Output classes: {n_classes}")
        
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 2e-2,
        patience: int = 20,
        verbose: bool = True
    ) -> dict:
        """
        Trenuje model z early stopping.
        
        Early stopping: zatrzyma trening jeśli walidacyjna metrika nie poprawia się
        przez 'patience' epok. Zapobiega overfittingowi na małych zbiorach danych.
        
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Update learning rate
        self.model.optimizer_params = dict(lr=learning_rate)
        
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],          # Metryka do early stopping
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=self.virtual_batch_size,
            weights=1,                    
            drop_last=False
        )
        
        print("\nTrening zakończony")
        
        return self._get_training_history()
    
    def _get_training_history(self) -> dict:
        """
        Zwraca historię treningu z modelu.
        """
        history = {
            'loss': [],
            'val_loss': [],
            'val_auc': []
        }
        if hasattr(self.model, 'history'):
            history = self.model.history
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja klasy (0 lub 1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja prawdopodobieństw.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, X: np.ndarray, feature_names: list = None) -> pd.DataFrame:
        """
        Zwraca ważność cech (feature importance).
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
                X = X.values
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif isinstance(X, pd.DataFrame):
             X = X.values

        masks = self.model.explain(X)
        importance = masks[0].mean(axis=0) 
        
        # Normalizujemy do [0, 1]
        importance = importance / importance.sum()
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Zapisuje model do pliku.
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        
        self.model.save_model(filepath)
        print(f"Model zapisany do: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Wczytuje model z pliku.
        """
        self.model = TabNetClassifier()
        self.model.load_model(filepath)
        print(f"Model wczytany z: {filepath}")
    


def evaluate_model(model: TabNetStrokeModel, X: np.ndarray, y: np.ndarray, dataset_name: str = "Dataset") -> dict:
    """
    Ewaluuje model na zbiorze danych.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Prawdopodobieństwo klasy 1
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics