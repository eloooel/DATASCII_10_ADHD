"""
Merge Peking_1/2/3 into a single "Peking" site in feature_manifest.csv
This enables proper LOSO validation without Peking_1's 0 ADHD problem.
"""

import pandas as pd
import shutil
from pathlib import Path

def merge_peking_sites():
    manifest_path = Path('data/features/feature_manifest.csv')
    backup_path = Path('data/features/feature_manifest_backup.csv')
    
    # Create backup
    shutil.copy(manifest_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"\nOriginal sites: {sorted(df['site'].unique())}")
    
    # Merge Peking_1/2/3 into "Peking"
    df.loc[df['site'].isin(['Peking_1', 'Peking_2', 'Peking_3']), 'site'] = 'Peking'
    
    # Save updated manifest
    df.to_csv(manifest_path, index=False)
    print(f"\nUpdated sites: {sorted(df['site'].unique())}")
    
    # Show distribution
    sites = ['NYU', 'Peking', 'NeuroIMAGE', 'KKI', 'OHSU']
    df_filtered = df[df['site'].isin(sites)]
    
    print(f"\n{'='*60}")
    print("Updated Distribution (5 Sites for LOSO):")
    print(f"{'='*60}")
    
    for site in sites:
        site_df = df_filtered[df_filtered['site'] == site]
        site_hc = (site_df['diagnosis'] == 0).sum()
        site_adhd = (site_df['diagnosis'] == 1).sum()
        total = len(site_df)
        print(f"  {site:12s}: Total={total:3d}, HC={site_hc:3d} ({site_hc/total*100:.1f}%), ADHD={site_adhd:3d} ({site_adhd/total*100:.1f}%)")
    
    total = len(df_filtered)
    hc = (df_filtered['diagnosis'] == 0).sum()
    adhd = (df_filtered['diagnosis'] == 1).sum()
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':12s}: Total={total:3d}, HC={hc:3d} ({hc/total*100:.1f}%), ADHD={adhd:3d} ({adhd/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    print("✓ Peking sites merged successfully!")
    print(f"✓ Updated manifest saved: {manifest_path}")
    print(f"\nTo restore original: mv {backup_path} {manifest_path}")

if __name__ == '__main__':
    merge_peking_sites()
