import pandas as pd, os, glob, sys

base = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2'

files = [
    ('MRI_CSV', os.path.join(base, 'MRI_Data_T1Sagittal3d3Tes.csv')),
    ('DXSUM', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_DXSUM_24Feb2026.csv')),
    ('Key_MRI', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_Key_MRI_24Feb2026.csv')),
    ('Key_PET', os.path.join(base, 'Tables_24Feb2026', 'All_Subjects_Key_PET_24Feb2026.csv')),
]

for name, f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        print(f'=== {name} ({df.shape[0]} rows x {df.shape[1]} cols) ===')
        print(f'File: {f}')
        print('Columns:', list(df.columns))
        print('First row:')
        print(df.iloc[0].to_dict())
        print()
    except Exception as e:
        print(f'ERROR {name}: {e}')
        print()

pet_dir = os.path.join(base, 'Pet18FFDG1f18_IDA_Metadata')
mri_dir = os.path.join(base, 'MRI_DATA_T1Sagittal3d3test_IDA_Metadata')
for d, name in [(pet_dir, 'PET_IDA'), (mri_dir, 'MRI_IDA')]:
    xmls = glob.glob(os.path.join(d, '**', '*.xml'), recursive=True)
    print(f'{name}: {len(xmls)} XML files')
    if xmls:
        print(f'  Sample XML path: {xmls[0]}')
        with open(xmls[0], 'r', encoding='utf-8', errors='replace') as fh:
            content = fh.read(3000)
        print('  XML content (first 3000 chars):')
        print(content)
        print()
