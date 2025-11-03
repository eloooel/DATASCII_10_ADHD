"""
Consolidated diagnostic utilities for WashU dataset analysis
Usage: python debug/washu_diagnostics.py [--structure|--status|--filter|--all]
"""
import pandas as pd
from pathlib import Path
import argparse


def check_washu_structure():
    """Analyze WashU data structure - multiple runs per subject"""
    METADATA_PATH = Path("data/raw/subjects_metadata.csv")
    
    print("="*70)
    print("WashU DATA STRUCTURE ANALYSIS")
    print("="*70)

    metadata = pd.read_csv(METADATA_PATH)
    washu = metadata[metadata['site'] == 'WashU']

    print(f"\n📊 Total WashU entries in metadata: {len(washu)}")
    print(f"📊 Unique WashU subjects: {washu['subject_id'].nunique()}")
    print(f"📊 Runs per subject: {len(washu) / washu['subject_id'].nunique():.1f} average")

    print("\n" + "="*70)
    print("RUNS PER SUBJECT BREAKDOWN")
    print("="*70)

    runs_per_subject = washu.groupby('subject_id').size()
    print(f"\nDistribution of runs per subject:")
    print(runs_per_subject.value_counts().sort_index())

    print("\n" + "="*70)
    print("SAMPLE: First 3 subjects with their runs")
    print("="*70)

    for subject_id in washu['subject_id'].unique()[:3]:
        subject_runs = washu[washu['subject_id'] == subject_id]
        print(f"\n{subject_id}: {len(subject_runs)} runs")
        for _, row in subject_runs.iterrows():
            filename = Path(row['input_path']).name
            print(f"  - Session {row.get('session', 1)}, Run {row['run']}: {filename}")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("For preprocessing, you should:")
    print("  1. Use only run-1 for each subject (60 subjects) - CURRENT APPROACH")
    print("  2. OR concatenate all runs per subject (more data per subject)")
    print("\nThis will give you 60 WashU subjects for analysis.")
    print("="*70)


def debug_filtering():
    """Debug WashU metadata filtering issues"""
    metadata = pd.read_csv('data/raw/subjects_metadata.csv')
    washu = metadata[metadata['site'] == 'WashU']

    print("="*70)
    print("WashU FILTERING DEBUG")
    print("="*70)
    
    print(f"\nTotal WashU entries: {len(washu)}")
    print(f"Unique WashU subjects: {washu['subject_id'].nunique()}")

    # The run column might be read as string or int
    print(f"\nRun column data type: {washu['run'].dtype}")
    print(f"Run values: {sorted(washu['run'].unique())}")
    
    # Check session distribution
    if 'session' in washu.columns:
        print(f"\nSession column data type: {washu['session'].dtype}")
        print(f"Session values: {sorted(washu['session'].unique())}")

    # Filter to run == 1 (handle both int and string)
    washu_run1 = washu[washu['run'] == 1]
    print(f"\n✅ WashU entries with run==1: {len(washu_run1)}")
    print(f"✅ Unique subjects with run==1: {washu_run1['subject_id'].nunique()}")

    # Check for multi-session subjects
    if 'session' in washu.columns:
        multi_session = washu_run1.groupby('subject_id')['session'].nunique()
        multi_session_subjects = multi_session[multi_session > 1]
        if len(multi_session_subjects) > 0:
            print(f"\n⚠️  {len(multi_session_subjects)} subjects have multiple sessions with run-1:")
            for subject_id in multi_session_subjects.index[:5]:
                sessions = washu_run1[washu_run1['subject_id'] == subject_id]['session'].tolist()
                print(f"    {subject_id}: sessions {sessions}")


def check_preprocessing_status():
    """Check preprocessing completion status across all sites"""
    print("="*70)
    print("PREPROCESSING STATUS CHECK")
    print("="*70)

    preprocessed_dir = Path('data/preprocessed')

    if not preprocessed_dir.exists():
        print("\n❌ No preprocessing directory found")
        return
    
    # Check for manifest files
    manifests = list(preprocessed_dir.glob('*/preprocessing_manifest.csv'))
    
    if not manifests:
        print("\n❌ No preprocessing manifests found")
        
        # Check for actual preprocessed directories
        sites = [d for d in preprocessed_dir.iterdir() if d.is_dir()]
        if sites:
            print(f"\n📁 Found {len(sites)} site directories:")
            for site_dir in sites:
                subjects = list(site_dir.glob('sub-*'))
                print(f"  {site_dir.name}: {len(subjects)} subject directories")
    else:
        total = 0
        complete = 0
        
        for manifest_path in manifests:
            df = pd.read_csv(manifest_path)
            site = manifest_path.parent.name
            comp = len(df[df['status']=='complete'])
            
            total += len(df)
            complete += comp
            
            print(f"\n{site}:")
            print(f"  Complete: {comp}/{len(df)}")
            print(f"  Failed: {len(df)-comp}")
        
        print(f"\n{'='*70}")
        print(f"TOTAL: {complete}/{total} subjects preprocessed")
        print(f"Success rate: {100*complete/total:.1f}%" if total > 0 else "N/A")
        
        if complete > 0:
            print(f"\n✅ Ready for feature extraction!")
        else:
            print(f"\n❌ No successfully preprocessed subjects")


def main():
    parser = argparse.ArgumentParser(
        description="WashU diagnostic utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug/washu_diagnostics.py --structure    # Analyze WashU data structure
  python debug/washu_diagnostics.py --status       # Check preprocessing status
  python debug/washu_diagnostics.py --filter       # Debug filtering issues
  python debug/washu_diagnostics.py --all          # Run all diagnostics
        """
    )
    
    parser.add_argument('--structure', action='store_true',
                       help='Analyze WashU multi-run structure')
    parser.add_argument('--status', action='store_true',
                       help='Check preprocessing completion status')
    parser.add_argument('--filter', action='store_true',
                       help='Debug metadata filtering issues')
    parser.add_argument('--all', action='store_true',
                       help='Run all diagnostic checks')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not (args.structure or args.status or args.filter or args.all):
        parser.print_help()
        return
    
    if args.all or args.structure:
        check_washu_structure()
        print("\n")
    
    if args.all or args.filter:
        debug_filtering()
        print("\n")
    
    if args.all or args.status:
        check_preprocessing_status()
        print("\n")


if __name__ == "__main__":
    main()
