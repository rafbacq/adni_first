import rdata, os

rda_dir = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2\ADNIMERGE2\build\data'
out_file = r'c:\Users\tdc65\adni\rda_inspect.txt'
lines = []

def read_rda(path):
    parsed = rdata.read_rda(path)
    result = rdata.conversion.convert(parsed)
    return result

# Check ADAS.rda
r = read_rda(os.path.join(rda_dir, 'ADAS.rda'))
print('ADAS keys:', list(r.keys()))
for k, v in r.items():
    if hasattr(v, 'columns'):
        print(f'{k}: shape={v.shape}')
        print('Cols:', list(v.columns))
        adas13_cols = [c for c in v.columns if 'ADAS13' in c.upper() or 'TOTMODTOTAL' in c.upper()]
        print('ADAS13-related cols:', adas13_cols)
print()

# Now scan all rda files for ADAS13
print('Scanning all RDA files for ADAS13...')
for fname in sorted(os.listdir(rda_dir)):
    if not fname.endswith('.rda'):
        continue
    try:
        r = read_rda(os.path.join(rda_dir, fname))
        for k, v in r.items():
            if hasattr(v, 'columns'):
                if any('ADAS13' in c.upper() for c in v.columns):
                    print(f'=== {fname}/{k} has ADAS13! shape={v.shape} ===')
                    print('Cols:', list(v.columns))
                    print()
    except Exception as e:
        pass  # skip unsupported
