import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional
import json

class DataSplitter:
    """Handles dataset splitting for model training, including train/test split and cross-validation"""
    
    def __init__(
        self,
        train_size: float = 0.8,
        n_splits: int = 5,
        random_state: int = 42,
        stratify: bool = True
    ):
        self.train_size = train_size
        self.test_size = 1 - train_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify = stratify
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
    def split_dataset(self, features_path: Path, splits_dir: Path) -> Dict:
        """Create and save train/test splits and CV folds"""
        # Load feature data
        data = pd.read_csv(features_path)
        
        # Get stratification labels if enabled
        stratify = data['diagnosis'] if self.stratify else None
        
        # Create train/test split
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Generate CV folds from training data
        cv_splits = []
        if self.n_splits > 1:
            for fold, (fold_train, fold_val) in enumerate(self.cv.split(
                data.iloc[train_idx], data.iloc[train_idx]['diagnosis']
            )):
                cv_splits.append({
                    'fold': fold + 1,
                    'train_idx': fold_train.tolist(),
                    'val_idx': fold_val.tolist()
                })
        
        # Prepare splits dictionary
        splits = {
            'train_idx': train_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'cv_splits': cv_splits,
            'metadata': {
                'n_samples': len(data),
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'n_folds': self.n_splits,
                'train_size': self.train_size,
                'random_state': self.random_state,
                'stratified': self.stratify
            }
        }
        
        # Save splits to disk
        splits_dir.mkdir(parents=True, exist_ok=True)
        with open(splits_dir / 'splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
            
        # Print summary
        print("\nDataset split complete:")
        print(f"Total samples: {len(data)}")
        print(f"Training set: {len(train_idx)} samples ({self.train_size*100:.1f}%)")
        print(f"Test set: {len(test_idx)} samples ({self.test_size*100:.1f}%)")
        if cv_splits:
            print(f"Cross-validation: {self.n_splits} folds")
            
        return splits
    
    def load_splits(self, splits_path: Path) -> Dict:
        """Load existing splits from disk"""
        with open(splits_path, 'r') as f:
            return json.load(f)
    
    def get_fold_data(
        self,
        data: pd.DataFrame,
        fold: Dict[str, List[int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract training and validation data for a specific CV fold"""
        train_data = data.iloc[fold['train_idx']]
        val_data = data.iloc[fold['val_idx']]
        return train_data, val_data