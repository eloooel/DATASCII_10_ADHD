"""
ROI Ranking Feature Selection Module
Based on SCCNN-RNN methodology for identifying most discriminative ROIs

Reference: Original baseline study used ROI-ranking to identify which of the 116 AAL regions
were most informative for ADHD classification. The same approach applies to Schaefer-200:
1. Extract features from all 200 ROIs
2. Train 200 separate models (one per ROI) with LOSO CV
3. Rank ROIs by their individual classification accuracy
4. Incrementally combine top-ranked ROIs to find optimal subset

This feature selection is crucial because:
- Reduces dimensionality
- Identifies brain regions most relevant for ADHD
- Improves model interpretability
- Can prevent overfitting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ROIRankingSelector:
    """
    Implements ROI-ranking feature selection strategy
    
    Process:
    1. First Round: Train model using each ROI independently
    2. Rank ROIs by their individual classification accuracy
    3. Second Round: Incrementally add top-ranked ROIs
    4. Identify optimal ROI subset based on performance curve
    """
    
    def __init__(
        self,
        n_rois: int = 200,
        validation_strategy: str = 'loso',  # 'loso' or 'cv'
        device: str = None
    ):
        self.n_rois = n_rois
        self.validation_strategy = validation_strategy
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.roi_rankings = None
        self.incremental_results = None
        self.optimal_rois = None
        
    def rank_rois_by_individual_performance(
        self,
        feature_manifest: pd.DataFrame,
        output_dir: Path,
        site_column: str = 'site'
    ) -> pd.DataFrame:
        """
        First Round: Evaluate each ROI independently
        
        Args:
            feature_manifest: DataFrame with columns [subject_id, site, fc_path, ts_path, diagnosis]
            output_dir: Directory to save ranking results
            site_column: Column name containing site information
            
        Returns:
            DataFrame with ROI rankings
        """
        print("\n" + "="*70)
        print("PHASE 1: ROI-RANKING - INDIVIDUAL ROI EVALUATION")
        print("="*70)
        print(f"Evaluating {self.n_rois} ROIs independently using {self.validation_strategy.upper()} validation")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all ROI timeseries data
        roi_data_dict = {}
        labels = []
        sites = []
        subject_ids = []
        
        print("\n📊 Loading ROI timeseries data...")
        for idx, row in tqdm(feature_manifest.iterrows(), total=len(feature_manifest), desc="Loading data"):
            ts_df = pd.read_csv(row['ts_path'])
            # Shape: (n_timepoints, n_rois)
            roi_data_dict[row['subject_id']] = ts_df.values
            labels.append(row['diagnosis'])
            sites.append(row[site_column])
            subject_ids.append(row['subject_id'])
        
        labels = np.array(labels)
        sites = np.array(sites)
        
        # Extract features per ROI and evaluate
        roi_results = []
        
        print(f"\n🎯 Evaluating each of {self.n_rois} ROIs individually...")
        for roi_idx in tqdm(range(self.n_rois), desc="ROI Evaluation"):
            # Extract this ROI's timeseries for all subjects (variable length)
            roi_timeseries_list = [
                roi_data_dict[subj][:, roi_idx] for subj in subject_ids
            ]  # List of arrays with different lengths
            
            # Compute simple statistical features (handles variable length)
            roi_features = self._compute_roi_features_from_list(roi_timeseries_list)
            
            # Evaluate with LOSO or CV
            if self.validation_strategy == 'loso':
                accuracy = self._evaluate_with_loso(roi_features, labels, sites)
            else:
                accuracy = self._evaluate_with_cv(roi_features, labels, n_folds=5)
            
            roi_results.append({
                'roi_id': roi_idx + 1,
                'roi_name': f'ROI_{roi_idx + 1}',
                'accuracy': accuracy
            })
        
        # Create rankings DataFrame
        rankings_df = pd.DataFrame(roi_results)
        rankings_df = rankings_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        # Save rankings
        rankings_path = output_dir / 'roi_rankings.csv'
        rankings_df.to_csv(rankings_path, index=False)
        
        # Save top 50 for quick reference
        top_50_path = output_dir / 'top_50_rois.csv'
        rankings_df.head(50).to_csv(top_50_path, index=False)
        
        print(f"\n✅ ROI rankings saved to: {rankings_path}")
        print(f"\nTop 15 ROIs:")
        print(rankings_df.head(15)[['rank', 'roi_name', 'accuracy']].to_string(index=False))
        
        self.roi_rankings = rankings_df
        return rankings_df
    
    def evaluate_incremental_roi_combinations(
        self,
        feature_manifest: pd.DataFrame,
        roi_rankings: pd.DataFrame,
        output_dir: Path,
        max_rois: int = 50,
        step_size: int = 1,
        site_column: str = 'site'
    ) -> pd.DataFrame:
        """
        Second Round: Incrementally combine top-ranked ROIs
        
        Tests combinations: top 1, top 2, top 3, ..., top N
        to find optimal number of ROIs
        
        Args:
            feature_manifest: DataFrame with feature paths
            roi_rankings: ROI rankings from first round
            output_dir: Directory to save results
            max_rois: Maximum number of ROIs to test
            step_size: ROI increment step (1 = test every combination, 5 = test every 5)
            site_column: Column name for site information
            
        Returns:
            DataFrame with incremental results
        """
        print("\n" + "="*70)
        print("PHASE 2: ROI-RANKING - INCREMENTAL COMBINATION EVALUATION")
        print("="*70)
        print(f"Testing combinations from top 1 to top {max_rois} ROIs")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        roi_data_dict = {}
        labels = []
        sites = []
        subject_ids = []
        
        print("\n📊 Loading ROI timeseries data...")
        for idx, row in tqdm(feature_manifest.iterrows(), total=len(feature_manifest), desc="Loading data"):
            ts_df = pd.read_csv(row['ts_path'])
            roi_data_dict[row['subject_id']] = ts_df.values
            labels.append(row['diagnosis'])
            sites.append(row[site_column])
            subject_ids.append(row['subject_id'])
        
        labels = np.array(labels)
        sites = np.array(sites)
        
        # Get ordered ROI indices by rank
        ranked_roi_indices = (roi_rankings.sort_values('rank')['roi_id'].values - 1).astype(int)
        
        # Test incremental combinations
        incremental_results = []
        
        roi_counts = list(range(1, max_rois + 1, step_size))
        
        print(f"\n🔬 Testing {len(roi_counts)} ROI combinations...")
        for n_rois in tqdm(roi_counts, desc="Incremental Evaluation"):
            # Select top N ROIs
            selected_roi_indices = ranked_roi_indices[:n_rois]
            
            # Extract features for selected ROIs (variable length timeseries)
            combined_timeseries_list = [
                roi_data_dict[subj][:, selected_roi_indices] for subj in subject_ids
            ]  # List of arrays with shape (n_timepoints, n_selected_rois)
            
            # Compute features (handles variable length)
            combined_features = self._compute_multi_roi_features_from_list(combined_timeseries_list)
            
            # Evaluate
            if self.validation_strategy == 'loso':
                accuracy = self._evaluate_with_loso(combined_features, labels, sites)
            else:
                accuracy = self._evaluate_with_cv(combined_features, labels, n_folds=5)
            
            incremental_results.append({
                'n_rois': n_rois,
                'selected_rois': selected_roi_indices.tolist(),
                'accuracy': accuracy
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(incremental_results)
        
        # Find optimal number of ROIs
        optimal_idx = results_df['accuracy'].idxmax()
        optimal_n_rois = results_df.loc[optimal_idx, 'n_rois']
        optimal_accuracy = results_df.loc[optimal_idx, 'accuracy']
        
        print(f"\n✅ Optimal ROI subset: Top {optimal_n_rois} ROIs")
        print(f"   Accuracy: {optimal_accuracy:.4f} ({optimal_accuracy*100:.2f}%)")
        
        # Save results
        results_path = output_dir / 'incremental_roi_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Save optimal ROI list
        optimal_rois = ranked_roi_indices[:optimal_n_rois]
        optimal_rois_df = roi_rankings[roi_rankings['roi_id'].isin(optimal_rois + 1)]
        optimal_path = output_dir / f'optimal_rois_top_{optimal_n_rois}.csv'
        optimal_rois_df.to_csv(optimal_path, index=False)
        
        print(f"\n📁 Results saved:")
        print(f"   Incremental results: {results_path}")
        print(f"   Optimal ROIs: {optimal_path}")
        
        # Plot performance curve
        self._plot_incremental_performance(results_df, output_dir, optimal_n_rois)
        
        self.incremental_results = results_df
        self.optimal_rois = optimal_rois
        
        return results_df
    
    def _compute_roi_features(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute statistical features from ROI timeseries
        
        Args:
            timeseries: Shape (n_subjects, n_timepoints)
            
        Returns:
            features: Shape (n_subjects, n_features)
        """
        features = []
        
        for ts in timeseries:
            feat = [
                np.mean(ts),           # Mean activation
                np.std(ts),            # Standard deviation
                np.min(ts),            # Min activation
                np.max(ts),            # Max activation
                np.median(ts),         # Median
                np.percentile(ts, 25), # 25th percentile
                np.percentile(ts, 75)  # 75th percentile
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _compute_roi_features_from_list(self, timeseries_list: list) -> np.ndarray:
        """
        Compute statistical features from variable-length ROI timeseries
        
        Args:
            timeseries_list: List of arrays with shape (n_timepoints,) each
            
        Returns:
            features: Shape (n_subjects, n_features)
        """
        features = []
        
        for ts in timeseries_list:
            feat = [
                np.mean(ts),           # Mean activation
                np.std(ts),            # Standard deviation
                np.min(ts),            # Min activation
                np.max(ts),            # Max activation
                np.median(ts),         # Median
                np.percentile(ts, 25), # 25th percentile
                np.percentile(ts, 75)  # 75th percentile
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _compute_multi_roi_features(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute features from multiple ROIs
        
        Args:
            timeseries: Shape (n_subjects, n_timepoints, n_rois)
            
        Returns:
            features: Shape (n_subjects, n_features)
        """
        n_subjects, n_timepoints, n_rois = timeseries.shape
        features = []
        
        for subj_ts in timeseries:
            # Shape: (n_timepoints, n_rois)
            subj_features = []
            
            # Statistical features per ROI
            for roi_idx in range(n_rois):
                roi_ts = subj_ts[:, roi_idx]
                subj_features.extend([
                    np.mean(roi_ts),
                    np.std(roi_ts),
                    np.max(roi_ts) - np.min(roi_ts)  # Range
                ])
            
            # Inter-ROI correlations
            if n_rois > 1:
                corr_matrix = np.corrcoef(subj_ts.T)
                upper_tri = corr_matrix[np.triu_indices(n_rois, k=1)]
                subj_features.extend([
                    np.mean(upper_tri),
                    np.std(upper_tri)
                ])
            
            features.append(subj_features)
        
        return np.array(features)
    
    def _compute_multi_roi_features_from_list(self, timeseries_list: list) -> np.ndarray:
        """
        Compute features from multiple ROIs with variable-length timeseries
        
        Args:
            timeseries_list: List of arrays with shape (n_timepoints, n_rois)
            
        Returns:
            features: Shape (n_subjects, n_features)
        """
        features = []
        
        for subj_ts in timeseries_list:
            # Shape: (n_timepoints, n_rois)
            n_timepoints, n_rois = subj_ts.shape
            subj_features = []
            
            # Statistical features per ROI
            for roi_idx in range(n_rois):
                roi_ts = subj_ts[:, roi_idx]
                subj_features.extend([
                    np.mean(roi_ts),
                    np.std(roi_ts),
                    np.max(roi_ts) - np.min(roi_ts)  # Range
                ])
            
            # Inter-ROI correlations
            if n_rois > 1:
                corr_matrix = np.corrcoef(subj_ts.T)
                upper_tri = corr_matrix[np.triu_indices(n_rois, k=1)]
                subj_features.extend([
                    np.mean(upper_tri),
                    np.std(upper_tri)
                ])
            
            features.append(subj_features)
        
        return np.array(features)
    
    def _evaluate_with_loso(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sites: np.ndarray
    ) -> float:
        """Evaluate using Leave-One-Site-Out cross-validation"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        logo = LeaveOneGroupOut()
        accuracies = []
        
        for train_idx, test_idx in logo.split(features, labels, sites):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Simple classifier for ranking
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _evaluate_with_cv(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 5
    ) -> float:
        """Evaluate using K-Fold cross-validation"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies = []
        
        for train_idx, test_idx in skf.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _plot_incremental_performance(
        self,
        results_df: pd.DataFrame,
        output_dir: Path,
        optimal_n: int
    ):
        """Plot ROI count vs accuracy curve"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(results_df['n_rois'], results_df['accuracy'] * 100, 
                marker='o', linewidth=2, markersize=4)
        plt.axvline(optimal_n, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_n} ROIs')
        plt.axhline(results_df['accuracy'].max() * 100, color='green', 
                   linestyle=':', alpha=0.5, label=f'Max Acc: {results_df["accuracy"].max()*100:.2f}%')
        
        plt.xlabel('Number of Top-Ranked ROIs', fontsize=12)
        plt.ylabel('LOSO Accuracy (%)', fontsize=12)
        plt.title('Incremental ROI Combination Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        plot_path = output_dir / 'incremental_roi_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Performance curve: {plot_path}")
    
    def create_filtered_feature_manifest(
        self,
        original_manifest: pd.DataFrame,
        optimal_rois: np.ndarray,
        output_path: Path
    ) -> pd.DataFrame:
        """
        Create new feature manifest with only optimal ROIs
        
        This is used for final model training
        """
        filtered_manifest = original_manifest.copy()
        filtered_manifest['selected_rois'] = [optimal_rois.tolist()] * len(filtered_manifest)
        filtered_manifest['n_selected_rois'] = len(optimal_rois)
        
        filtered_manifest.to_csv(output_path, index=False)
        
        print(f"\n✅ Filtered manifest created: {output_path}")
        print(f"   Using {len(optimal_rois)} ROIs for final training")
        
        return filtered_manifest


def run_roi_ranking_pipeline(
    feature_manifest_path: Path,
    output_dir: Path,
    n_rois: int = 200,
    max_rois_to_test: int = 50,
    validation_strategy: str = 'loso'
) -> Dict:
    """
    Complete ROI-ranking pipeline
    
    Args:
        feature_manifest_path: Path to feature manifest CSV
        output_dir: Directory for ranking results
        n_rois: Total number of ROIs in atlas
        max_rois_to_test: Maximum ROIs to test in incremental phase
        validation_strategy: 'loso' or 'cv'
        
    Returns:
        Dictionary with ranking results and optimal ROI configuration
    """
    print("\n" + "="*70)
    print("🧠 STARTING ROI-RANKING FEATURE SELECTION PIPELINE")
    print("="*70)
    
    # Load feature manifest
    feature_manifest = pd.read_csv(feature_manifest_path)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total subjects: {len(feature_manifest)}")
    print(f"   Sites: {feature_manifest['site'].nunique()}")
    print(f"   Total ROIs: {n_rois}")
    print(f"   Validation: {validation_strategy.upper()}")
    
    # Initialize selector
    selector = ROIRankingSelector(
        n_rois=n_rois,
        validation_strategy=validation_strategy
    )
    
    # Phase 1: Rank individual ROIs
    roi_rankings = selector.rank_rois_by_individual_performance(
        feature_manifest=feature_manifest,
        output_dir=output_dir
    )
    
    # Phase 2: Incremental combination
    incremental_results = selector.evaluate_incremental_roi_combinations(
        feature_manifest=feature_manifest,
        roi_rankings=roi_rankings,
        output_dir=output_dir,
        max_rois=max_rois_to_test,
        step_size=1  # Test every combination for precision
    )
    
    # Create filtered manifest for final training
    optimal_rois = selector.optimal_rois
    filtered_manifest_path = output_dir / 'feature_manifest_optimal_rois.csv'
    filtered_manifest = selector.create_filtered_feature_manifest(
        original_manifest=feature_manifest,
        optimal_rois=optimal_rois,
        output_path=filtered_manifest_path
    )
    
    # Summary
    results = {
        'roi_rankings': roi_rankings,
        'incremental_results': incremental_results,
        'optimal_n_rois': len(optimal_rois),
        'optimal_rois': optimal_rois.tolist(),
        'optimal_accuracy': incremental_results['accuracy'].max(),
        'filtered_manifest_path': str(filtered_manifest_path)
    }
    
    # Save summary
    summary_path = output_dir / 'roi_ranking_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'optimal_n_rois': results['optimal_n_rois'],
            'optimal_accuracy': float(results['optimal_accuracy']),
            'optimal_roi_ids': [int(x + 1) for x in results['optimal_rois']],
            'validation_strategy': validation_strategy,
            'n_total_rois': n_rois
        }, f, indent=2)
    
    print(f"\n✅ ROI-Ranking pipeline complete!")
    print(f"   Summary saved to: {summary_path}")
    
    return results
