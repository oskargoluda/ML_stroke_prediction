"""
przygotowywanie danych dla modelu 

Zawiera:
-wczytuje surowe dane z pliku CSV
-przeprowadza analizę eksploracyjną danych (EDA)
-czyści dane (obsługa brakujących wartości, usuwanie niespójności)
-tworzy nowe cechy (feature engineering)
-dzieli dane na zbiory treningowy, walidacyjny i testowy z zachowaniem proporcji 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrokeDataPreparator:
    def __init__(self, raw_data_path=None, processed_data_path=None, random_state=42):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path or "data/processed"
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        logger.info(f"wczytuje dane z pliku:{self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        return df
    
    def analyze_data(self, df):
        return {
            'shape': df.shape,
            'missing': df.isnull().sum().to_dict(),
            'target_dist': df['stroke'].value_counts().to_dict()
        }
    
    def clean_and_preprocess(self, df):
        df=df.copy()
        
        # usuniecie kolumny id
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Dropped 'id' column")
        
        df = df.dropna(subset=['bmi'])
        
        if (df['age'] == 0).any():
            df = df[df['age'] > 0]
        logger.info(f"Data shape after cleaning: {df.shape}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Tworzy nowe cechy na podstawie istniejących danych.
        """
        df = df.copy()
        
        #Grupy wiekowe
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                                 labels=['child', 'young_adult', 'middle_aged', 'senior', 'elderly'])
        logger.info("Created 'age_group' feature")
        
        # BMI 
        df['bmi_category'] = pd.cut(df['bmi'], 
                                    bins=[0, 18.5, 25, 30, 1000],
                                    labels=['underweight', 'normal', 'overweight', 'obese'])
        logger.info("Created 'bmi_category' feature")
        
        # Kategorie poziomu glukozy
        df['glucose_level_category'] = pd.cut(df['avg_glucose_level'],
                                              bins=[0, 100, 125, 200, 10000],
                                              labels=['normal', 'impaired_fasting', 'diabetic', 'very_high'])
        logger.info("Created 'glucose_level_category' feature")
        
        # Liczba czynników ryzyka (jak jest nadciśnienie, choroby serca, palenie to +1)
        df['risk_factor_count'] = (
                                   df['hypertension'] + 
                                   df['heart_disease'] + 
                                   (df['smoking_status'] != 'never smoked').astype(int))
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        przekształca cechy kategoryczne na numeryczne za pomocą Label Encoding.
        """
        
        df = df.copy()
        #cechy 
        categorical_features = ['gender', 'work_type', 'Residence_type', 'smoking_status',
                                'age_group', 'bmi_category', 'glucose_level_category', 'ever_married']
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
                    logger.info(f"Encoded '{feature}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
                else:
                    le = self.label_encoders[feature]
                    df[feature] = le.transform(df[feature].astype(str))
        
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """
        Skaluje cechy numeryczne za pomocą StandardScaler.
        """
        df = df.copy()
        
        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        if fit:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            logger.info(f"Fitted scaler on {len(numerical_features)} numerical features")
        else:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    def prepare_for_modeling(self, df):
        """
        Przygotowuje dane do modelowania, oddzielając cechy od celu.
        """

        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def split_data(self, X, y, val_size=0.15, test_size=0.15):
        """
        Dzieli dane na zbiory treningowy, walidacyjny i testowy w proporcjach 70%-train,15%-val,15%-test.
        """
        logger.info("\n=== SPLITTING DATA ===")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=self.random_state, stratify=y_temp
        )
        
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        zapisuje przetworzone dane do plików CSV.
        """
        
        X_train.to_csv(f"{self.processed_data_path}/X_train.csv", index=False)
        X_val.to_csv(f"{self.processed_data_path}/X_val.csv", index=False)
        X_test.to_csv(f"{self.processed_data_path}/X_test.csv", index=False)
        y_train.to_csv(f"{self.processed_data_path}/y_train.csv", index=False)
        y_val.to_csv(f"{self.processed_data_path}/y_val.csv", index=False)
        y_test.to_csv(f"{self.processed_data_path}/y_test.csv", index=False)
        
    
    def save_preprocessing_info(self, filepath):
        info = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            #'numerical_features': ['age', 'avg_glucose_level', 'bmi']
            'numerical_features': ['age_log', 'avg_glucose_level', 'bmi']
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(info, f)

    def load_preprocessing_info(self, filepath):
        """
        wczytuje informacje o preprocessingu z pliku.
        """
        with open(filepath, 'rb') as f:
            info = pickle.load(f)
        
        self.label_encoders = info['label_encoders']
        self.scaler = info['scaler']

    def transform(self, df):
        """
        Przekształcanie danych przy pomocy napisanych funkcji
        """
        df = df.copy()
        df = self.clean_and_preprocess(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_numerical_features(df, fit=False)
        
        if self.feature_names:
            df = df[self.feature_names]
            
        return df

    def prepare(self, data_path=None, save_data=True):
        """
        Pełny pipeline przygotowania danych.
        1. Wczytuje dane
        2. Analizuje dane
        3. Czyści i preprocessuje dane
        4. Tworzy cechy
        5. Koduje cechy kategoryczne
        6. Skaluje cechy numeryczne
        7. Przygotowuje dane do modelowania
        8. Dzieli dane na zbiory
        9. Zapisuje przetworzone dane (opcjonalnie)
        """
        if data_path:
            self.raw_data_path = data_path
            
        if not self.raw_data_path:
            raise ValueError("No data path provided (neither in init nor in prepare)")
            
        logger.info("Starting data preparation pipeline...")
        
        #1
        df = self.load_data()
        #2
        self.analyze_data(df)
        #3
        df = self.clean_and_preprocess(df)
        #4
        df = self.feature_engineering(df)
        #5
        df = self.encode_categorical_features(df, fit=True)
        #6
        df = self.scale_numerical_features(df, fit=True)
        #7
        X, y = self.prepare_for_modeling(df)
        #8
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        #9
        if save_data:
            self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logger.info("Data preparation pipeline completed!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    dp = StrokeDataPreparator("data/raw/healthcare-dataset-stroke-data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = dp.prepare()
