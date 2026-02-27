import os, json
from pathlib import Path
import pandas as pd

out = Path(r'c:\Users\tdc65\adni\outputs')
seq = out / 'sequences'

files = {
    'Merged CSV': out / 'adni_spine_adas13_multimodal.csv',
    'Column summary': out / 'column_summary.md',
    'Splits JSON': out / 'splits.json',
    'Sequences manifest': seq / 'manifest.csv',
}
for name, p in files.items():
    size = p.stat().st_size if p.exists() else -1
    print(f'{name}: {"OK" if p.exists() else "MISSING"} ({size:,} bytes)')

npz_count = len(list(seq.glob('**/*.npz')))
print(f'Sequence .npz files: {npz_count}')

with open(out/'splits.json') as f:
    splits = json.load(f)
for k, v in splits.items():
    print(f'  {k}: {len(v)} subjects')

mf = pd.read_csv(seq/'manifest.csv')
print(f'Manifest: {mf.shape[0]} rows')
print(mf.groupby('split').agg({'n_visits':'mean','n_features':'first','adas13_obs':'mean'}).round(1))
