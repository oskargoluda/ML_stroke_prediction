import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import warnings
warnings.filterwarnings('ignore')


class TabNetStrokeModel:
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
        mask_type: str = "sparsemax",
        lambda_sparse: float = 1e-3
    ) -> TabNetClassifier:
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
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
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
        history = {
            'loss': [],
            'val_loss': [],
            'val_auc': []
        }
        if hasattr(self.model, 'history'):
            history = self.model.history
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        
        self.model.save_model(filepath)
        print(f"Model zapisany do: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        self.model = TabNetClassifier()
        self.model.load_model(filepath)
        print(f"Model wczytany z: {filepath}")
    


def evaluate_model(model: TabNetStrokeModel, X: np.ndarray, y: np.ndarray, dataset_name: str = "Dataset") -> dict:

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