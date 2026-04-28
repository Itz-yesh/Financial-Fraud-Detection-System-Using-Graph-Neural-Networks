"""
Data Preprocessor Module
Feature engineering, normalization, and train/val/test splitting
Implements class imbalance analysis (Tong et al. 2023)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocesses transaction data for GNN training"""
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize preprocessor
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, val, and test ratios must sum to 1.0"
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit preprocessor and transform data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed DataFrame, preprocessing metadata)
        """
        df = df.copy()
        
        # Identify column types
        self._identify_column_types(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df, fit=True)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Normalize numerical features
        df = self._normalize_features(df, fit=True)
        
        # Analyze class imbalance (Tong et al. 2023)
        imbalance_info = self._analyze_class_imbalance(df)
        
        metadata = {
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'imbalance_info': imbalance_info
        }
        
        return df, metadata
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df, fit=False)
        df = self._engineer_features(df)
        df = self._normalize_features(df, fit=False)
        
        return df
    
    def _identify_column_types(self, df: pd.DataFrame):
        """Identify categorical and numerical columns"""
        # Exclude label and ID columns
        exclude_cols = ['is_fraud', 'transaction_id', 'user_id', 'merchant_id', 
                       'device_id', 'timestamp']
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_columns.append(col)
            elif np.issubdtype(df[col].dtype, np.number):
                self.numerical_columns.append(col)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns: fill with median
        for col in self.numerical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns: fill with 'unknown'
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna('unknown')

        # For ID columns: fill with -1
        id_cols = ['user_id', 'merchant_id', 'device_id']
        for col in id_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(-1)

        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing ones
        """
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            if np.issubdtype(df['timestamp'].dtype, np.number):
                # PaySim: 'step' is integer hours since simulation start
                # Derive cyclic time features directly from the hour offset
                step = df['timestamp']
                df['hour'] = (step % 24).astype(int)
                df['day_of_week'] = ((step // 24) % 7).astype(int)
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            else:
                # Datetime string (synthetic / IEEE-CIS datasets)
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Add to numerical columns
            for col in ['hour', 'day_of_week', 'is_weekend']:
                if col not in self.numerical_columns:
                    self.numerical_columns.append(col)

        # Amount-based features if amount exists
        if 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'])

            # Amount bins — note: bins are exclusive on left, so amount=0 falls outside
            # bin (0, 10] and becomes NaN. Fill with 0 (own bucket) before int cast.
            df['amount_bin'] = pd.cut(df['amount'],
                                     bins=[-np.inf, 0, 10, 50, 100, 500, 1000, np.inf],
                                     labels=[0, 1, 2, 3, 4, 5, 6])
            df['amount_bin'] = df['amount_bin'].fillna(0).astype(int)
            
            if 'log_amount' not in self.numerical_columns:
                self.numerical_columns.append('log_amount')
            if 'amount_bin' not in self.numerical_columns:
                self.numerical_columns.append('amount_bin')
        
        return df

    
    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features"""
        if not self.numerical_columns:
            return df
        
        # Get columns that exist in dataframe
        cols_to_normalize = [col for col in self.numerical_columns if col in df.columns]
        
        if not cols_to_normalize:
            return df
        
        if fit:
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        else:
            df[cols_to_normalize] = self.scaler.transform(df[cols_to_normalize])
        
        # Store feature columns
        self.feature_columns = cols_to_normalize
        
        return df
    
    def _analyze_class_imbalance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze class imbalance in the dataset (Tong et al. 2023)
        
        Args:
            df: DataFrame with 'is_fraud' column
            
        Returns:
            Dictionary with imbalance statistics
        """
        if 'is_fraud' not in df.columns:
            return {}
        
        fraud_count = df['is_fraud'].sum()
        normal_count = len(df) - fraud_count
        fraud_ratio = fraud_count / len(df)
        imbalance_ratio = normal_count / fraud_count if fraud_count > 0 else float('inf')
        
        info = {
            'total_samples': len(df),
            'fraud_samples': int(fraud_count),
            'normal_samples': int(normal_count),
            'fraud_ratio': float(fraud_ratio),
            'imbalance_ratio': float(imbalance_ratio),
            'is_severely_imbalanced': imbalance_ratio > 10,
            'recommended_pos_weight': float(imbalance_ratio)
        }
        
        print("\n" + "="*50)
        print("CLASS IMBALANCE ANALYSIS (Tong et al. 2023)")
        print("="*50)
        print(f"Total samples: {info['total_samples']:,}")
        print(f"Fraudulent: {info['fraud_samples']:,} ({fraud_ratio*100:.2f}%)")
        print(f"Normal: {info['normal_samples']:,} ({(1-fraud_ratio)*100:.2f}%)")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"Severely imbalanced: {'YES' if info['is_severely_imbalanced'] else 'NO'}")
        print(f"Recommended positive class weight: {info['recommended_pos_weight']:.2f}")
        print("="*50 + "\n")
        
        return info
    
    def split_data(self, 
                   df: pd.DataFrame,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            stratify: Whether to stratify split by fraud label
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        stratify_col = df['is_fraud'] if stratify and 'is_fraud' in df.columns else None
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=stratify_col
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        stratify_col_train_val = train_val_df['is_fraud'] if stratify and 'is_fraud' in train_val_df.columns else None
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=stratify_col_train_val
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df


def preprocess_data(df: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple:
    """
    Convenience function to preprocess and split data
    
    Args:
        df: Input DataFrame
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        test_ratio: Test data ratio
        
    Returns:
        Tuple of (train_df, val_df, test_df, preprocessor, metadata)
    """
    preprocessor = DataPreprocessor(train_ratio, val_ratio, test_ratio)
    
    # Fit and transform
    df_processed, metadata = preprocessor.fit_transform(df)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df_processed)
    
    return train_df, val_df, test_df, preprocessor, metadata
