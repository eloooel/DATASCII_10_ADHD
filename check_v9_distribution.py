import pandas as pd

df = pd.read_csv('data/features/feature_manifest.csv')

sites = ['Brown', 'KKI', 'NYU', 'NeuroIMAGE', 'OHSU', 'Peking', 'Pittsburgh', 'WashU']
df_filtered = df[df['site'].isin(sites)]

print("="*80)
print("V9 DATASET DISTRIBUTION (ALL 8 SITES)")
print("="*80)
print(f"\nTotal subjects: {len(df_filtered)}")
print(f"HC (Control):   {(df_filtered['diagnosis'] == 0).sum()} ({(df_filtered['diagnosis'] == 0).sum()/len(df_filtered)*100:.1f}%)")
print(f"ADHD:           {(df_filtered['diagnosis'] == 1).sum()} ({(df_filtered['diagnosis'] == 1).sum()/len(df_filtered)*100:.1f}%)")

print("\n" + "="*80)
print("PER-SITE DISTRIBUTION (Each site becomes test set once in LOSO)")
print("="*80)
print(f"{'Site':<15} {'HC':>5} {'ADHD':>5} {'Total':>6} {'ADHD %':>8}")
print("-"*80)

for site in sites:
    site_df = df_filtered[df_filtered['site'] == site]
    hc = (site_df['diagnosis'] == 0).sum()
    adhd = (site_df['diagnosis'] == 1).sum()
    total = len(site_df)
    adhd_pct = (adhd / total * 100) if total > 0 else 0
    print(f"{site:<15} {hc:5d} {adhd:5d} {total:6d} {adhd_pct:7.1f}%")

print("="*80)

print("\n" + "="*80)
print("LOSO TEST FOLD BREAKDOWN")
print("="*80)
print("\nIn 8-fold LOSO cross-validation:")
print("- Each site is held out as test set once")
print("- Remaining 7 sites are used for training")
print("\nTest fold sizes:")

for i, site in enumerate(sites, 1):
    site_df = df_filtered[df_filtered['site'] == site]
    hc = (site_df['diagnosis'] == 0).sum()
    adhd = (site_df['diagnosis'] == 1).sum()
    train_df = df_filtered[df_filtered['site'] != site]
    train_hc = (train_df['diagnosis'] == 0).sum()
    train_adhd = (train_df['diagnosis'] == 1).sum()
    
    print(f"\nFold {i} - Test: {site}")
    print(f"  Test:  {hc:3d} HC + {adhd:3d} ADHD = {hc+adhd:3d} total")
    print(f"  Train: {train_hc:3d} HC + {train_adhd:3d} ADHD = {train_hc+train_adhd:3d} total")
