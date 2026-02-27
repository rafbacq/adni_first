"""
Explore R package RDA files by scanning for ADAS13 data.
R ADAM-style packages store data as PROMISE with ALTREP - we need to
understand the lazy evaluation wrapper and extract inner data.
Strategy: Try every .rda that might have ADAS13 by scanning binary content.
"""
import bz2, gzip, lzma, os, re

rda_dir = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2\ADNIMERGE2\build\data'

def decompress(path):
    with open(path, 'rb') as f:
        head = f.read(2)
    if head[:2] == b'BZ':
        with bz2.open(path, 'rb') as f:
            return f.read()
    elif head[:2] == b'\x1f\x8b':
        with gzip.open(path, 'rb') as f:
            return f.read()
    else:
        try:
            with lzma.open(path, 'rb') as f:
                return f.read()
        except:
            with open(path, 'rb') as f:
                return f.read()

results = {}
for fname in sorted(os.listdir(rda_dir)):
    if not fname.endswith('.rda'):
        continue
    try:
        data = decompress(os.path.join(rda_dir, fname))
        # Scan for 'ADAS13' as ASCII string in the binary data
        content_latin = data.decode('latin-1', errors='replace')
        # Look for ADAS13 exactly
        indices = [m.start() for m in re.finditer(r'ADAS13', content_latin)]
        if indices:
            results[fname] = {'adas13_locs': indices}
            print(f'{fname}: ADAS13 found at {indices[:5]}')
            # Show context
            for idx in indices[:2]:
                ctx = content_latin[max(0,idx-20):idx+50]
                print(f'  Context: {repr(ctx)}')
            print()
        
        # Also look for column name 'ADAS13' in any context
        if 'ADAS1' in content_latin and fname not in results:
            print(f'{fname}: contains ADAS1*')
    except Exception as e:
        pass

print('\nFiles with ADAS13:', list(results.keys()))
