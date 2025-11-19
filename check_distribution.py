import pandas as pd

df = pd.read_csv('data/features/feature_manifest.csv')
sites = ['NYU', 'Peking_1', 'Peking_2', 'Peking_3', 'NeuroIMAGE', 'KKI', 'OHSU']
df_filtered = df[df['site'].isin(sites)]

total = len(df_filtered)
hc = (df_filtered['diagnosis'] == 0).sum()
adhd = (df_filtered['diagnosis'] == 1).sum()

print(f'7 Baseline Sites:')
print(f'  Total={total}')
print(f'  HC={hc} ({hc/total*100:.1f}%)')
print(f'  ADHD={adhd} ({adhd/total*100:.1f}%)')
print(f'  Ratio: {hc}:{adhd} = {hc/adhd:.2f}:1')
print(f'\nPer-site breakdown:')

for site in sites:
    site_df = df_filtered[df_filtered['site'] == site]
    site_hc = (site_df['diagnosis'] == 0).sum()
    site_adhd = (site_df['diagnosis'] == 1).sum()
    print(f'  {site:12s}: Total={len(site_df):3d}, HC={site_hc:3d}, ADHD={site_adhd:3d}')
