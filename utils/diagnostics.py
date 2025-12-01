"""
Diagnostic and Utility Scripts for ADHD Pipeline

Common utilities for:
- Checking subject status
- Regenerating manifests
- Reprocessing failed subjects
- Analyzing dataset statistics
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def check_missing_subjects(metadata_path='data/raw/subjects_metadata.csv',
                           feature_dir='data/features'):
    """
    Identify subjects in metadata but missing from features
    
    Args:
        metadata_path: Path to subjects metadata CSV
        feature_dir: Directory containing feature files
    """
    print("="*70)
    print("CHECKING FOR MISSING SUBJECTS")
    print("="*70)
    
    metadata = pd.read_csv(metadata_path)
    feature_dir = Path(feature_dir)
    
    missing_subjects = []
    
    for _, row in metadata.iterrows():
        subject_id = row['subject_id']
        site = row.get('site', row.get('dataset', 'unknown'))
        
        fc_path = feature_dir / site / f"{subject_id}_connectivity_matrix.npy"
        ts_path = feature_dir / site / f"{subject_id}_roi_timeseries.csv"
        
        if not (fc_path.exists() and ts_path.exists()):
            missing_subjects.append({
                'subject_id': subject_id,
                'site': site,
                'diagnosis': row.get('diagnosis', row.get('DX', -1)),
                'fc_exists': fc_path.exists(),
                'ts_exists': ts_path.exists()
            })
    
    if missing_subjects:
        print(f"\nWarning: Found {len(missing_subjects)} missing subjects:")
        df = pd.DataFrame(missing_subjects)
        print(df.to_string(index=False))
        
        print(f"\nBy diagnosis:")
        print(df['diagnosis'].value_counts())
    else:
        print("\nSuccess: All subjects in metadata have features!")
    
    return missing_subjects


def regenerate_feature_manifest(metadata_path='data/raw/subjects_metadata.csv',
                                feature_dir='data/features',
                                output_path=None):
    """
    Regenerate feature_manifest.csv with corrected diagnosis labels
    
    Args:
        metadata_path: Path to subjects metadata CSV
        feature_dir: Directory containing feature files
        output_path: Output path for manifest (default: feature_dir/feature_manifest.csv)
    """
    print("="*70)
    print("REGENERATING FEATURE MANIFEST")
    print("="*70)
    
    metadata = pd.read_csv(metadata_path)
    feature_dir = Path(feature_dir)
    
    if output_path is None:
        output_path = feature_dir / 'feature_manifest.csv'
    
    manifest_data = []
    missing_count = 0
    
    for _, row in metadata.iterrows():
        subject_id = row['subject_id']
        site = row.get('site', row.get('dataset', 'unknown'))
        
        fc_path = feature_dir / site / f"{subject_id}_connectivity_matrix.npy"
        ts_path = feature_dir / site / f"{subject_id}_roi_timeseries.csv"
        
        if fc_path.exists() and ts_path.exists():
            manifest_data.append({
                'subject_id': subject_id,
                'site': site,
                'fc_path': str(fc_path),
                'ts_path': str(ts_path),
                'diagnosis': row.get('diagnosis', row.get('DX', 0))
            })
        else:
            missing_count += 1
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(output_path, index=False)
    
    print(f"\nSummary: Feature Manifest Updated:")
    print(f"  Complete: {len(manifest_df)} subjects")
    print(f"  Missing: {missing_count} subjects")
    print(f"\n📈 Diagnosis Distribution:")
    print(manifest_df['diagnosis'].value_counts().to_dict())
    print(f"\n🌍 By Site:")
    site_dx = pd.crosstab(manifest_df['site'], manifest_df['diagnosis'], margins=True)
    print(site_dx)
    print(f"\n💾 Saved to: {output_path}")
    
    return manifest_df


def analyze_dataset_stats(manifest_path='data/features/feature_manifest.csv'):
    """
    Print comprehensive dataset statistics
    
    Args:
        manifest_path: Path to feature manifest CSV
    """
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    df = pd.read_csv(manifest_path)
    
    total = len(df)
    adhd = (df['diagnosis'] == 1).sum()
    control = (df['diagnosis'] == 0).sum()
    
    print(f"\nSummary: Overall:")
    print(f"  Total subjects: {total}")
    print(f"  ADHD: {adhd} ({100*adhd/total:.1f}%)")
    print(f"  Control: {control} ({100*control/total:.1f}%)")
    
    print(f"\nBy Site:")
    site_stats = df.groupby('site')['diagnosis'].agg([
        ('total', 'count'),
        ('ADHD', lambda x: (x==1).sum()),
        ('Control', lambda x: (x==0).sum())
    ])
    print(site_stats)
    
    print(f"\nClass Balance:")
    print(f"  ADHD/Total ratio: {adhd/total:.3f}")
    print(f"  Imbalance ratio: {max(adhd, control)/min(adhd, control):.2f}:1")
    
    return df


def retry_failed_subjects(subject_ids, manifest_path='data/raw/subjects_metadata.csv',
                         output_dir='data/preprocessed'):
    """
    Retry preprocessing for specific failed subjects
    
    Args:
        subject_ids: List of subject IDs to retry
        manifest_path: Path to subjects metadata CSV
        output_dir: Output directory for preprocessing
    """
    from preprocessing.preprocess import _process_subject
    import gc
    
    print("="*70)
    print("RETRYING FAILED SUBJECTS")
    print("="*70)
    
    manifest_df = pd.read_csv(manifest_path)
    subject_ids = list(set(subject_ids))  # Remove duplicates
    
    print(f"\nTarget subjects ({len(subject_ids)}):")
    for subj_id in subject_ids:
        print(f"   - {subj_id}")
    
    results = []
    
    for i, subj_id in enumerate(subject_ids, 1):
        match = manifest_df[manifest_df['subject_id'] == subj_id]
        
        if match.empty:
            print(f"\n[{i}/{len(subject_ids)}] Not in metadata: {subj_id}")
            continue
        
        row = match.iloc[0].to_dict()
        row['out_dir'] = output_dir
        site = row.get('site', 'unknown')
        
        print(f"\n[{i}/{len(subject_ids)}] Processing {subj_id} ({site})")
        
        try:
            result = _process_subject(row)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"   SUCCESS")
            else:
                print(f"   FAILED: {result.get('error', 'Unknown')[:60]}")
        except Exception as e:
            print(f"   EXCEPTION: {str(e)[:60]}")
            results.append({
                'status': 'failed',
                'subject_id': subj_id,
                'error': str(e)
            })
        
        gc.collect()
    
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary: Results: {success}/{len(results)} successful")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ADHD Pipeline Utilities")
    parser.add_argument('action', choices=['check-missing', 'regenerate-manifest', 'stats', 'retry'],
                       help="Action to perform")
    parser.add_argument('--subjects', nargs='+', help="Subject IDs for retry action")
    
    args = parser.parse_args()
    
    if args.action == 'check-missing':
        check_missing_subjects()
    
    elif args.action == 'regenerate-manifest':
        regenerate_feature_manifest()
    
    elif args.action == 'stats':
        analyze_dataset_stats()
    
    elif args.action == 'retry':
        if not args.subjects:
            print("Error: --subjects required for retry action")
            sys.exit(1)
        retry_failed_subjects(args.subjects)
