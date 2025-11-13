"""
Generate a comprehensive table showing metadata and preprocessing status for each site
"""
import pandas as pd
from pathlib import Path

# Load metadata
meta_path = Path("data/raw/subjects_metadata.csv")
if not meta_path.exists():
    print("❌ Metadata file not found")
    exit(1)

meta = pd.read_csv(meta_path)

# Load integrity report
report_path = Path("data/preprocessed/integrity_report.csv")
if not report_path.exists():
    print("❌ Integrity report not found. Run scripts/check_preprocessed_integrity.py first.")
    exit(1)

integrity = pd.read_csv(report_path)

# Create summary table
summary_data = []

for site in sorted(meta['site'].unique()):
    site_meta = meta[meta['site'] == site]
    site_integrity = integrity[integrity['site'] == site]
    
    # Metadata stats
    total_runs = len(site_meta)
    unique_subjects = site_meta['subject_id'].nunique()
    avg_runs = total_runs / unique_subjects if unique_subjects > 0 else 0
    
    # Preprocessing stats
    total_subjects_in_report = len(site_integrity)
    valid_preprocessed = len(site_integrity[site_integrity['preproc_valid'] == True])
    invalid_preprocessed = total_subjects_in_report - valid_preprocessed
    
    # Percentage
    pct_complete = (valid_preprocessed / unique_subjects * 100) if unique_subjects > 0 else 0
    
    summary_data.append({
        'Site': site,
        'Total_Runs': total_runs,
        'Unique_Subjects': unique_subjects,
        'Avg_Runs_Per_Subject': round(avg_runs, 2),
        'Preprocessed_Valid': valid_preprocessed,
        'Preprocessed_Invalid': invalid_preprocessed,
        'Preprocessing_Complete_%': round(pct_complete, 1)
    })

# Create DataFrame
summary_df = pd.DataFrame(summary_data)

# Add totals row
totals = {
    'Site': 'TOTAL',
    'Total_Runs': summary_df['Total_Runs'].sum(),
    'Unique_Subjects': summary_df['Unique_Subjects'].sum(),
    'Avg_Runs_Per_Subject': round(summary_df['Total_Runs'].sum() / summary_df['Unique_Subjects'].sum(), 2),
    'Preprocessed_Valid': summary_df['Preprocessed_Valid'].sum(),
    'Preprocessed_Invalid': summary_df['Preprocessed_Invalid'].sum(),
    'Preprocessing_Complete_%': round(summary_df['Preprocessed_Valid'].sum() / summary_df['Unique_Subjects'].sum() * 100, 1)
}
summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)

# Display table
print("="*100)
print("SITE SUMMARY TABLE - Metadata & Preprocessing Status")
print("="*100)
print()
print(summary_df.to_string(index=False))
print()
print("="*100)
print("\nLegend:")
print("  Total_Runs: Number of fMRI run files in metadata")
print("  Unique_Subjects: Number of unique subject IDs")
print("  Avg_Runs_Per_Subject: Average number of runs per subject")
print("  Preprocessed_Valid: Number of subjects with valid preprocessed outputs")
print("  Preprocessed_Invalid: Number of subjects missing or with invalid preprocessed outputs")
print("  Preprocessing_Complete_%: Percentage of subjects successfully preprocessed")
print()

# Save to CSV
output_path = Path("data/preprocessed/site_summary.csv")
summary_df.to_csv(output_path, index=False)
print(f"✅ Summary saved to: {output_path}")
