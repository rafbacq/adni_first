import pandas as pd, os, glob, sys

base = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2'
out = r'c:\Users\tdc65\adni\schema_log.txt'

lines = []

files = [
    ('MRI_CSV', os.path.join(base, 'MRI_Data_T1Sagittal3d3Tes.csv')),
    ('DXSUM', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_DXSUM_24Feb2026.csv')),
    ('Key_MRI', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_Key_MRI_24Feb2026.csv')),
    ('Key_PET', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_Key_PET_24Feb2026.csv')),
]

for name, f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        lines.append(f'=== {name} ({df.shape[0]} rows x {df.shape[1]} cols) ===')
        lines.append(f'File: {f}')
        lines.append(f'Columns: {list(df.columns)}')
        lines.append(f'DTypes: {df.dtypes.to_dict()}')
        lines.append(f'First row:\n{df.iloc[0].to_string()}')
        if name == 'Key_PET':
            # Check unique descriptions / tracers
            desc_cols = [c for c in df.columns if 'desc' in c.lower() or 'tracer' in c.lower() or 'radio' in c.lower() or 'modality' in c.lower()]
            lines.append(f'PET description-like cols: {desc_cols}')
            for col in desc_cols[:3]:
                lines.append(f'  Unique {col}: {df[col].value_counts().head(20).to_dict()}')
        lines.append('')
    except Exception as e:
        lines.append(f'ERROR {name}: {e}')
        lines.append('')

# IDA XML structure
pet_dir = os.path.join(base, 'Pet18FFDG1f18_IDA_Metadata')
mri_dir = os.path.join(base, 'MRI_DATA_T1Sagittal3d3test_IDA_Metadata')
for d, name in [(pet_dir, 'PET_IDA'), (mri_dir, 'MRI_IDA')]:
    xmls = glob.glob(os.path.join(d, '**', '*.xml'), recursive=True)
    lines.append(f'{name}: {len(xmls)} XML files')
    if xmls:
        lines.append(f'  Sample XML path: {xmls[0]}')
        try:
            with open(xmls[0], 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read(4000)
            lines.append('  XML content (first 4000 chars):')
            lines.append(content)
        except Exception as e:
            lines.append(f'  Error reading XML: {e}')
    lines.append('')

# Check RDA to see if there's ADNIMERGE csv elsewhere
rda_dir = os.path.join(base, 'ADNIMERGE2', 'build', 'data')
rda_files = [f for f in os.listdir(rda_dir) if 'ADNIMERGE' in f.upper()]
lines.append(f'RDA files with ADNIMERGE in name: {rda_files}')
all_rdas = os.listdir(rda_dir)
lines.append(f'Total RDA files: {len(all_rdas)}')
lines.append(f'Some RDA names: {all_rdas[:30]}')

with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Written to {out}')
