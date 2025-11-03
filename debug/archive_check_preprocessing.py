from pathlib import Path
import pandas as pd

print("="*60)
print("PREPROCESSING STATUS CHECK")
print("="*60)

preprocessed_dir = Path('data/preprocessed')

if not preprocessed_dir.exists():
    print("\n❌ No preprocessing directory found")
else:
    manifests = list(preprocessed_dir.glob('*/preprocessing_manifest.csv'))
    
    if not manifests:
        print("\n❌ No preprocessing manifests found")
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
        
        print(f"\n{'='*60}")
        print(f"TOTAL: {complete}/{total} subjects preprocessed")
        print(f"Success rate: {100*complete/total:.1f}%" if total > 0 else "N/A")
        
        if complete > 0:
            print(f"\n✅ Ready for feature extraction!")
        else:
            print(f"\n❌ No successfully preprocessed subjects")
