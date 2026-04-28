"""
Data Loader Module
Loads financial fraud datasets (IEEE-CIS, Elliptic, PaySim, or synthetic)
Handles data downloading, caching, and initial validation
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict
from sklearn.datasets import make_classification


class DataLoader:
    """Handles loading of various financial fraud datasets"""
    
    def __init__(self, 
                 dataset_name: str = "synthetic",
                 data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of dataset (synthetic, ieee-cis, elliptic, paysim)
            data_dir: Directory containing raw data files
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.supported_datasets = ['synthetic', 'ieee-cis', 'elliptic', 'paysim']
        
        if self.dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Choose from: {self.supported_datasets}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset based on dataset_name
        
        Returns:
            DataFrame containing transaction data
        """
        if self.dataset_name == "synthetic":
            return self._load_synthetic_data()
        elif self.dataset_name == "ieee-cis":
            return self._load_ieee_cis_data()
        elif self.dataset_name == "elliptic":
            return self._load_elliptic_data()
        elif self.dataset_name == "paysim":
            return self._load_paysim_data()
    
    def _load_synthetic_data(self, 
                            n_samples: int = 10000,
                            fraud_ratio: float = 0.05) -> pd.DataFrame:
        """
        Generate synthetic transaction data for testing
        
        Args:
            n_samples: Number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            DataFrame with synthetic transaction data
        """
        np.random.seed(42)
        
        # Generate base features using sklearn
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=2,
            weights=[1-fraud_ratio, fraud_ratio],
            flip_y=0.01,
            random_state=42
        )
        
        # Create realistic transaction features
        df = pd.DataFrame()
        
        # Transaction identifiers
        df['transaction_id'] = [f"TX_{i:06d}" for i in range(n_samples)]
        df['user_id'] = np.random.randint(0, n_samples // 5, n_samples)
        df['merchant_id'] = np.random.randint(0, n_samples // 10, n_samples)
        
        # Transaction amount (fraudulent transactions tend to be higher)
        base_amount = np.abs(np.random.lognormal(4, 1.5, n_samples))
        fraud_multiplier = np.where(y == 1, np.random.uniform(1.5, 3.0, n_samples), 1.0)
        df['amount'] = base_amount * fraud_multiplier
        
        # Timestamp (sequential with some randomness)
        base_time = pd.Timestamp('2024-01-01')
        time_deltas = np.cumsum(np.random.exponential(300, n_samples))  # seconds
        df['timestamp'] = [base_time + pd.Timedelta(seconds=int(td)) for td in time_deltas]
        
        # Transaction type
        transaction_types = ['purchase', 'transfer', 'withdrawal', 'payment']
        df['transaction_type'] = np.random.choice(transaction_types, n_samples)
        
        # Device and location features
        df['device_id'] = np.random.randint(0, n_samples // 3, n_samples)
        df['ip_address'] = [f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}" 
                           for _ in range(n_samples)]
        df['country'] = np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_samples)
        
        # Add sklearn-generated features
        for i in range(X.shape[1]):
            df[f'feature_{i}'] = X[:, i]
        
        # Label
        df['is_fraud'] = y
        
        # Add some missing values (realistic scenario)
        missing_mask = np.random.random(n_samples) < 0.02
        df.loc[missing_mask, 'device_id'] = np.nan
        
        print(f"Generated synthetic dataset:")
        print(f"  Total transactions: {n_samples}")
        print(f"  Fraudulent: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        print(f"  Normal: {len(y) - y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)")
        print(f"  Features: {len(df.columns)}")
        
        return df
    
    def _load_ieee_cis_data(self) -> pd.DataFrame:
        """
        Load IEEE-CIS Fraud Detection dataset
        
        Note: This dataset should be downloaded from Kaggle:
        https://www.kaggle.com/c/ieee-fraud-detection/data
        
        Returns:
            DataFrame with IEEE-CIS data
        """
        train_transaction_path = os.path.join(self.data_dir, "train_transaction.csv")
        
        if not os.path.exists(train_transaction_path):
            print(f"IEEE-CIS dataset not found at {train_transaction_path}")
            print("Please download from: https://www.kaggle.com/c/ieee-fraud-detection/data")
            print("Falling back to synthetic data...")
            return self._load_synthetic_data()
        
        # Load transaction data
        df = pd.read_csv(train_transaction_path)
        
        # Rename columns for consistency
        if 'isFraud' in df.columns:
            df.rename(columns={'isFraud': 'is_fraud'}, inplace=True)
        
        print(f"Loaded IEEE-CIS dataset:")
        print(f"  Total transactions: {len(df)}")
        print(f"  Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        
        return df
    
    def _load_elliptic_data(self) -> pd.DataFrame:
        """
        Load Elliptic Bitcoin Dataset
        
        Note: This dataset should be downloaded from:
        https://www.kaggle.com/ellipticco/elliptic-data-set
        
        Returns:
            DataFrame with Elliptic data
        """
        features_path = os.path.join(self.data_dir, "elliptic_txs_features.csv")
        classes_path = os.path.join(self.data_dir, "elliptic_txs_classes.csv")
        edges_path = os.path.join(self.data_dir, "elliptic_txs_edgelist.csv")
        
        if not all(os.path.exists(p) for p in [features_path, classes_path, edges_path]):
            print(f"Elliptic dataset not found in {self.data_dir}")
            print("Please download from: https://www.kaggle.com/ellipticco/elliptic-data-set")
            print("Falling back to synthetic data...")
            return self._load_synthetic_data()
        
        # Load features and classes
        features = pd.read_csv(features_path, header=None)
        classes = pd.read_csv(classes_path)
        
        # Merge
        df = features.merge(classes, left_on=0, right_on='txId', how='left')
        
        # Process labels (1=illicit, 2=licit, unknown=unlabeled)
        df['is_fraud'] = df['class'].apply(lambda x: 1 if x == '1' else 0 if x == '2' else -1)
        
        # Remove unlabeled transactions for supervised learning
        df = df[df['is_fraud'] != -1]
        
        print(f"Loaded Elliptic dataset:")
        print(f"  Total transactions: {len(df)}")
        print(f"  Illicit: {(df['is_fraud']==1).sum()} ({(df['is_fraud']==1).mean()*100:.2f}%)")
        
        return df
    
    def _load_paysim_data(self) -> pd.DataFrame:
        """
        Load PaySim synthetic financial dataset

        PaySim columns (original -> renamed):
          step        -> timestamp  (integer: hours since simulation start)
          nameOrig    -> user_id
          nameDest    -> merchant_id
          isFraud     -> is_fraud
          type        -> transaction_type  (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
          amount, oldbalanceOrg, newbalanceOrig,
          oldbalanceDest, newbalanceDest, isFlaggedFraud  -- kept as-is

        Note: This dataset should be downloaded from:
        https://www.kaggle.com/ntnu-testimon/paysim1
        Place the CSV in data/raw/ (filename starting with PS_)

        Returns:
            DataFrame with PaySim data
        """
        # Support any filename starting with PS_ so users don't need to rename
        import glob
        pattern = os.path.join(self.data_dir, "PS_*.csv")
        matches = glob.glob(pattern)

        if not matches:
            print(f"PaySim dataset not found in {self.data_dir} (expected a file matching PS_*.csv)")
            print("Please download from: https://www.kaggle.com/ntnu-testimon/paysim1")
            print("Falling back to synthetic data...")
            return self._load_synthetic_data()

        paysim_path = matches[0]  # use the first match
        print(f"Loading PaySim file: {paysim_path}")

        # Load data
        df = pd.read_csv(paysim_path)

        # Rename columns for pipeline consistency
        column_mapping = {
            'isFraud': 'is_fraud',
            'nameOrig': 'user_id',
            'nameDest': 'merchant_id',
            'step': 'timestamp',          # integer hours -- handled by preprocessor
            'type': 'transaction_type',
        }
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns},
                  inplace=True)

        # Add columns expected by preprocessor but absent in PaySim
        # transaction_id: unique row identifier
        df['transaction_id'] = [f"TX_{i:08d}" for i in range(len(df))]
        # device_id: not present in PaySim; set to NaN so missing-value handler fills it
        if 'device_id' not in df.columns:
            df['device_id'] = np.nan

        print(f"Loaded PaySim dataset:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.4f}%)")
        print(f"  Normal:     {(df['is_fraud'] == 0).sum():,}")

        # --- Stratified sampling -------------------------------------------
        # 6.3M rows is too large to build a graph in RAM.
        # Keep ALL fraud cases + sample remaining budget from normal rows so
        # the fraud/normal ratio is preserved as much as possible.
        max_rows = 200_000  # adjust higher if you have more RAM (e.g. 500_000)
        if len(df) > max_rows:
            fraud_df   = df[df['is_fraud'] == 1]
            normal_df  = df[df['is_fraud'] == 0]
            n_normal   = min(max_rows - len(fraud_df), len(normal_df))
            normal_sample = normal_df.sample(n=n_normal, random_state=42)
            df = pd.concat([fraud_df, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"\n  [Sampled to {len(df):,} rows for graph construction]")
            print(f"  Fraud kept: {df['is_fraud'].sum():,} | Normal sampled: {n_normal:,}")
        # -------------------------------------------------------------------

        return df
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get information about the dataset
        
        Returns:
            Dictionary with dataset metadata
        """
        df = self.load_data()
        
        info = {
            'dataset_name': self.dataset_name,
            'n_samples': len(df),
            'n_features': len(df.columns),
            'n_fraud': df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
            'fraud_ratio': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return info


def load_dataset(dataset_name: str = "synthetic", 
                data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Convenience function to load dataset
    
    Args:
        dataset_name: Name of dataset
        data_dir: Data directory
        
    Returns:
        DataFrame with transaction data
    """
    loader = DataLoader(dataset_name, data_dir)
    return loader.load_data()
