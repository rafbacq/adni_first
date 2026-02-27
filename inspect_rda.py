import pyreadr, os

rda_dir = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2\ADNIMERGE2\build\data'
out_file = r'c:\Users\tdc65\adni\rda_inspect.txt'
lines = []

# Check ADAS.rda
result = pyreadr.read_r(os.path.join(rda_dir, 'ADAS.rda'))
df = list(result.values())[0]
lines.append(f'ADAS shape: {df.shape}')
lines.append(f'ADAS cols: {list(df.columns)}')
adas13_cols = [c for c in df.columns if 'ADAS' in c.upper() or '13' in c or 'COG' in c.upper()]
lines.append(f'ADAS13-related cols: {adas13_cols}')
lines.append(f'First row:\n{df.iloc[0].to_string()}')
lines.append('')

# Check ADSL
result2 = pyreadr.read_r(os.path.join(rda_dir, 'ADSL.rda'))
df2 = list(result2.values())[0]
lines.append(f'ADSL shape: {df2.shape}')
lines.append(f'ADSL cols: {list(df2.columns)}')
lines.append('')

# Check ADQS (cognitive scores)
if os.path.exists(os.path.join(rda_dir, 'ADQS.rda')):
    r3 = pyreadr.read_r(os.path.join(rda_dir, 'ADQS.rda'))
    df3 = list(r3.values())[0]
    lines.append(f'ADQS shape: {df3.shape}')
    lines.append(f'ADQS cols: {list(df3.columns)}')
    adas13 = [c for c in df3.columns if 'ADAS13' in c.upper()]
    lines.append(f'ADAS13 cols in ADQS: {adas13}')
    lines.append('')

# Look for ADNIMERGE-like data in all rdas
for fname in os.listdir(rda_dir):
    if fname.endswith('.rda'):
        try:
            r = pyreadr.read_r(os.path.join(rda_dir, fname))
            df = list(r.values())[0]
            if 'ADAS13' in [c.upper() for c in df.columns] or any('ADAS13' in c.upper() for c in df.columns):
                lines.append(f'=== {fname} has ADAS13! shape={df.shape} ===')
                lines.append(f'Cols: {list(df.columns)}')
                lines.append('')
        except Exception as e:
            pass

with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Done')
