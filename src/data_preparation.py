"""
Data Preparation Module for Stroke Prediction

This module handles:
- Loading raw data
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Train/Val/Test split with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrokeDataPreparator:
    """
    Handles all data preparation tasks for stroke prediction model.
    
    Attributes:
        raw_data_path (str): Path to raw CSV file
        processed_data_path (str): Path to save processed data
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, raw_data_path=None, processed_data_path=None, random_state=42):
        """
        Initialize the data preparation pipeline.
        
        Args:
            raw_data_path (str, optional): Path to healthcare-dataset-stroke-data.csv
            processed_data_path (str): Directory to save processed data
            random_state (int): Seed for reproducibility
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path or "data/processed"
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """
        Load raw data from CSV.
        
        Returns:
            pd.DataFrame: Raw data
        """
        logger.info(f"Loading data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def analyze_data(self, df):
        """
        Print basic statistics and information about the data.
        
        Args:
            df (pd.DataFrame): Data to analyze
        """
        logger.info("\n=== DATA ANALYSIS ===")
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"\nData types:\n{df.dtypes}")
        logger.info(f"\nMissing values:\n{df.isnull().sum()}")
        logger.info(f"\nTarget variable distribution:\n{df['stroke'].value_counts()}")
        try:
            ratio = df['stroke'].value_counts()[0] / df['stroke'].value_counts()[1]
            logger.info(f"Target imbalance ratio: {ratio:.2f}:1")
        except:
            logger.info("Could not calculate imbalance ratio (possibly 0 count)")
            
        logger.info(f"\nBasic statistics:\n{df.describe()}")
        
        return {
            'shape': df.shape,
            'missing': df.isnull().sum().to_dict(),
            'target_dist': df['stroke'].value_counts().to_dict()
        }
    
    def clean_and_preprocess(self, df):
        """
        Clean data: handle missing values, remove inconsistencies.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("\n=== DATA CLEANING ===")
        
        # Create a copy
        df = df.copy()
        
        # Drop 'id' column - not useful for prediction
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Dropped 'id' column")
        
        # Handle missing values
        logger.info(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # BMI: remove rows with missing BMI (small number)
        # Alternative: could impute with median, but here we remove
        df = df.dropna(subset=['bmi'])
        logger.info(f"Removed rows with missing BMI. Shape now: {df.shape}")
        
        # smoking_status: fill 'Unknown' (leave as is, it's a valid category)
        
        # Handle age=0 (impossible values)
        if (df['age'] == 0).any():
            logger.warning(f"Found {(df['age'] == 0).sum()} rows with age=0, removing them")
            df = df[df['age'] > 0]
        
        logger.info(f"Data shape after cleaning: {df.shape}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Cleaned data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("\n=== FEATURE ENGINEERING ===")
        
        df = df.copy()
        
        # Age groups (discretization)
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                                 labels=['child', 'young_adult', 'middle_aged', 'senior', 'elderly'])
        logger.info("Created 'age_group' feature")
        
        # BMI categories (WHO standard)
        df['bmi_category'] = pd.cut(df['bmi'], 
                                    bins=[0, 18.5, 25, 30, 1000],
                                    labels=['underweight', 'normal', 'overweight', 'obese'])
        logger.info("Created 'bmi_category' feature")
        
        # Glucose level categories
        df['glucose_level_category'] = pd.cut(df['avg_glucose_level'],
                                              bins=[0, 100, 125, 200, 10000],
                                              labels=['normal', 'impaired_fasting', 'diabetic', 'very_high'])
        logger.info("Created 'glucose_level_category' feature")
        
        # Risk factors count
        df['risk_factor_count'] = (
                                   df['hypertension'] + 
                                   df['heart_disease'] + 
                                   (df['smoking_status'] != 'never smoked').astype(int))
        logger.info("Created 'risk_factor_count' feature")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features to numerical values.
        
        Args:
            df (pd.DataFrame): Data with categorical features
            fit (bool): If True, fit encoders; if False, use existing encoders
            
        Returns:
            pd.DataFrame: Data with encoded features
        """
        logger.info("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        df = df.copy()
        
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
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Data with numerical features
            fit (bool): If True, fit scaler; if False, use existing scaler
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        logger.info("\n=== SCALING NUMERICAL FEATURES ===")
        
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
        Final preparation: separate features and target, prepare for train/val/test split.
        
        Args:
            df (pd.DataFrame): Preprocessed data
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        logger.info("\n=== PREPARING FOR MODELING ===")
        
        # Separate features and target
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Feature names: {self.feature_names}")
        
        return X, y
    
    def split_data(self, X, y, val_size=0.15, test_size=0.15):
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            val_size (float): Validation set size relative to remaining data
            test_size (float): Test set size relative to original data
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("\n=== SPLITTING DATA ===")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape}, class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Val set: {X_val.shape}, class distribution: {y_val.value_counts().to_dict()}")
        logger.info(f"Test set: {X_test.shape}, class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Save processed data to disk.
        
        Args:
            X_train, X_val, X_test, y_train, y_val, y_test: Data splits
        """
        logger.info("\n=== SAVING PROCESSED DATA ===")
        
        X_train.to_csv(f"{self.processed_data_path}/X_train.csv", index=False)
        X_val.to_csv(f"{self.processed_data_path}/X_val.csv", index=False)
        X_test.to_csv(f"{self.processed_data_path}/X_test.csv", index=False)
        y_train.to_csv(f"{self.processed_data_path}/y_train.csv", index=False)
        y_val.to_csv(f"{self.processed_data_path}/y_val.csv", index=False)
        y_test.to_csv(f"{self.processed_data_path}/y_test.csv", index=False)
        
        logger.info(f"Saved all data splits to {self.processed_data_path}/")
    
    def save_preprocessing_info(self, filepath):
        """
        Save preprocessing artifacts (encoders, scalers) for inference.
        """
        info = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'numerical_features': ['age', 'avg_glucose_level', 'bmi']
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(info, f)
        
        logger.info(f"Saved preprocessing info to {filepath}")

    def load_preprocessing_info(self, filepath):
        """
        Load preprocessing artifacts (encoders, scalers) for inference.
        """
        with open(filepath, 'rb') as f:
            info = pickle.load(f)
        
        self.label_encoders = info['label_encoders']
        self.scaler = info['scaler']
        logger.info(f"Loaded preprocessing info from {filepath}")

    def transform(self, df):
        """
        Transform raw data for inference.
        """
        df = df.copy()
        df = self.clean_and_preprocess(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_numerical_features(df, fit=False)
        
        # Select features in correct order
        if self.feature_names:
            df = df[self.feature_names]
            
        return df

    def prepare(self, data_path=None, save_data=True):
        """
        Run the complete data preparation pipeline.
        
        Args:
            data_path (str, optional): Path to raw data. Overrides init path.
            save_data (bool): Whether to save processed data
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if data_path:
            self.raw_data_path = data_path
            
        if not self.raw_data_path:
            raise ValueError("No data path provided (neither in init nor in prepare)")
            
        logger.info("Starting data preparation pipeline...")
        
        # Load
        df = self.load_data()
        
        # Analyze
        self.analyze_data(df)
        
        # Clean
        df = self.clean_and_preprocess(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode
        df = self.encode_categorical_features(df, fit=True)
        
        # Scale
        df = self.scale_numerical_features(df, fit=True)
        
        # Prepare
        X, y = self.prepare_for_modeling(df)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Save
        if save_data:
            self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logger.info("Data preparation pipeline completed!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    dp = StrokeDataPreparator("data/raw/healthcare-dataset-stroke-data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = dp.prepare()
