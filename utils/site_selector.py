"""
Site Selection Utilities

Handles filtering datasets for different experimental configurations:
1. All 8 sites (759 subjects) - Maximum data
2. Baseline-comparable subset (5 sites) - Fair comparison with baseline study

Baseline study used: NYU, Peking (combined), OHSU, KKI, NI
Your dataset has: Brown, NYU, NeuroIMAGE, Peking_1, Peking_2, Peking_3, Pittsburgh, WashU

Mapping:
- NYU → NYU (direct match)
- Peking → Peking_1 + Peking_2 + Peking_3 (combined)
- OHSU → Not available (use alternative)
- KKI → Not available (use alternative)
- NI → NeuroIMAGE (likely match)

Alternative baseline-comparable configuration:
Use the 5 largest/most reliable sites from your dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class SiteSelector:
    """Manages site selection for different experimental configurations"""
    
    # Site definitions
    ALL_SITES = ['Brown', 'NYU', 'NeuroIMAGE', 'Peking_1', 'Peking_2', 'Peking_3', 'Pittsburgh', 'WashU', 'KKI', 'OHSU']
    
    # Baseline-comparable configuration
    # Matches baseline study: NYU, Peking (1+2+3), OHSU, KKI, NI (NeuroIMAGE)
    BASELINE_SITES = ['NYU', 'Peking_1', 'Peking_2', 'Peking_3', 'NeuroIMAGE', 'KKI', 'OHSU']
    
    # Alternative: use only sites with sufficient samples (>50 subjects)
    LARGE_SITES = ['NYU', 'Pittsburgh', 'NeuroIMAGE']
    
    def __init__(self, metadata_path: Path = None):
        """
        Initialize site selector
        
        Args:
            metadata_path: Path to subjects_metadata.csv (optional, for analysis)
        """
        self.metadata_path = metadata_path
        self.metadata = None
        
        if metadata_path and metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
    
    def get_site_statistics(self) -> pd.DataFrame:
        """Get statistics for all sites"""
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Provide metadata_path during initialization.")
        
        # Group by site and subject
        subject_counts = self.metadata.groupby(['site', 'subject_id']).size().reset_index(name='n_runs')
        
        stats = []
        for site in self.ALL_SITES:
            site_subjects = subject_counts[subject_counts['site'] == site]
            
            if len(site_subjects) == 0:
                continue
            
            # Get diagnosis info
            site_metadata = self.metadata[self.metadata['site'] == site]
            
            stats.append({
                'site': site,
                'n_subjects': len(site_subjects),
                'n_runs': site_subjects['n_runs'].sum(),
                'avg_runs_per_subject': site_subjects['n_runs'].mean(),
                'multi_run_subjects': (site_subjects['n_runs'] > 1).sum()
            })
        
        return pd.DataFrame(stats).sort_values('n_subjects', ascending=False)
    
    def filter_by_sites(
        self,
        data: pd.DataFrame,
        site_config: str = 'all',
        site_column: str = 'site'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filter dataset by site configuration
        
        Args:
            data: DataFrame with site information
            site_config: 'all', 'baseline', or 'large'
            site_column: Column name containing site information
            
        Returns:
            Filtered DataFrame and list of selected sites
        """
        if site_config == 'all':
            selected_sites = self.ALL_SITES
            config_name = "All 8 Sites"
        elif site_config == 'baseline':
            selected_sites = self.BASELINE_SITES
            config_name = "Baseline-Comparable (5 sites)"
        elif site_config == 'large':
            selected_sites = self.LARGE_SITES
            config_name = "Large Sites Only"
        else:
            raise ValueError(f"Unknown site_config: {site_config}")
        
        # Filter data
        filtered_data = data[data[site_column].isin(selected_sites)].copy()
        
        print(f"\nConfiguration: Site Configuration: {config_name}")
        print(f"   Selected sites: {', '.join(selected_sites)}")
        print(f"   Total subjects: {len(filtered_data)}")
        
        if 'diagnosis' in filtered_data.columns:
            n_adhd = (filtered_data['diagnosis'] == 1).sum()
            n_control = (filtered_data['diagnosis'] == 0).sum()
            print(f"   ADHD: {n_adhd}, Control: {n_control}")
        
        # Per-site breakdown
        site_counts = filtered_data.groupby(site_column).size().sort_values(ascending=False)
        print(f"\n   Per-site counts:")
        for site, count in site_counts.items():
            print(f"      {site}: {count}")
        
        return filtered_data, selected_sites
    
    def create_experiment_configs(
        self,
        feature_manifest: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Create separate feature manifests for each experimental configuration
        
        Args:
            feature_manifest: Original feature manifest with all subjects
            output_dir: Directory to save filtered manifests
            
        Returns:
            Dictionary mapping config name to manifest path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = {}
        
        # Config 1: All 8 sites
        all_sites_data, all_sites_list = self.filter_by_sites(
            feature_manifest,
            site_config='all'
        )
        all_sites_path = output_dir / 'feature_manifest_all_sites.csv'
        all_sites_data.to_csv(all_sites_path, index=False)
        configs['all_sites'] = all_sites_path
        
        # Config 2: Baseline-comparable
        baseline_data, baseline_sites_list = self.filter_by_sites(
            feature_manifest,
            site_config='baseline'
        )
        baseline_path = output_dir / 'feature_manifest_baseline_sites.csv'
        baseline_data.to_csv(baseline_path, index=False)
        configs['baseline'] = baseline_path
        
        # Save configuration info
        config_info = {
            'all_sites': {
                'sites': all_sites_list,
                'n_subjects': len(all_sites_data),
                'manifest_path': str(all_sites_path)
            },
            'baseline': {
                'sites': baseline_sites_list,
                'n_subjects': len(baseline_data),
                'manifest_path': str(baseline_path),
                'note': 'Comparable to baseline study site selection'
            }
        }
        
        config_info_path = output_dir / 'experiment_configurations.json'
        with open(config_info_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"\nSuccess: Experiment configurations created:")
        print(f"   All sites: {all_sites_path}")
        print(f"   Baseline: {baseline_path}")
        print(f"   Config info: {config_info_path}")
        
        return configs
    
    def validate_site_compatibility(self) -> Dict:
        """
        Check if dataset sites are suitable for baseline comparison
        
        Returns compatibility analysis
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded")
        
        stats = self.get_site_statistics()
        
        analysis = {
            'total_sites': len(stats),
            'baseline_sites_available': len([s for s in self.BASELINE_SITES if s in stats['site'].values]),
            'site_details': stats.to_dict('records'),
            'recommendations': []
        }
        
        # Check sample sizes
        for _, row in stats.iterrows():
            if row['n_subjects'] < 30:
                analysis['recommendations'].append(
                    f"⚠️ {row['site']}: Low sample size ({row['n_subjects']} subjects) - "
                    f"may affect generalizability"
                )
            
            if row['multi_run_subjects'] > 0:
                analysis['recommendations'].append(
                    f"✅ {row['site']}: {row['multi_run_subjects']} multi-run subjects - "
                    f"good for data quality"
                )
        
        return analysis


def create_site_filtered_datasets(
    feature_manifest_path: Path,
    metadata_path: Path,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Convenience function to create all site-filtered datasets
    
    Args:
        feature_manifest_path: Path to original feature manifest
        metadata_path: Path to subjects_metadata.csv
        output_dir: Output directory for filtered manifests
        
    Returns:
        Dictionary of configuration paths
    """
    print("\n" + "="*70)
    print("CREATING SITE-FILTERED EXPERIMENTAL CONFIGURATIONS")
    print("="*70)
    
    # Initialize selector
    selector = SiteSelector(metadata_path)
    
    # Validate site compatibility
    print("\n📋 Site Compatibility Analysis:")
    compatibility = selector.validate_site_compatibility()
    
    print(f"\n   Total sites: {compatibility['total_sites']}")
    print(f"   Baseline-compatible sites: {compatibility['baseline_sites_available']}/5")
    
    if compatibility['recommendations']:
        print("\n   Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"      {rec}")
    
    # Load feature manifest
    feature_manifest = pd.read_csv(feature_manifest_path)
    
    # Create configurations
    configs = selector.create_experiment_configs(
        feature_manifest=feature_manifest,
        output_dir=output_dir
    )
    
    print("\nSuccess: Site-filtered datasets ready for experiments")
    
    return configs
