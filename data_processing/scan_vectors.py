"""
Extract ADAS.rda data frame by direct binary scan of R XDR format.
Uses known column names and row count (12868) from R package documentation.
"""
import bz2, struct, os, math
import pandas as pd

def decompress(path):
    with open(path, 'rb') as f:
        head = f.read(2)
    if head[:2] == b'BZ':
        with bz2.open(path, 'rb') as f:
            return f.read()
    import gzip
    try:
        with gzip.open(path, 'rb') as f:
            return f.read()
    except:
        import lzma
        with lzma.open(path, 'rb') as f:
            return f.read()

def read_strsxp_at(data, offset):
    """Read a STRSXP at the given offset (offset points to flags)."""
    n = struct.unpack_from('>i', data, offset+4)[0]
    if n < 0 or n > 1000000:
        return None, offset
    vals = []
    pos = offset + 8
    for j in range(n):
        if pos + 8 > len(data):
            break
        flags = struct.unpack_from('>I', data, pos)[0]
        sxptype = flags & 0xff
        levels = (flags >> 12) & 0xfffff
        
        if sxptype == 9:  # CHARSXP
            slen = struct.unpack_from('>i', data, pos+4)[0]
            if slen == -1:
                vals.append(None)
                pos += 8
            elif 0 <= slen <= 100000:
                s = data[pos+8:pos+8+slen].decode('utf-8', errors='replace')
                vals.append(s)
                pos += 8 + slen
            else:
                return None, offset  # invalid
        elif sxptype == 254:  # REFSXP - back reference
            ref_idx = struct.unpack_from('>i', data, pos+4)[0]
            vals.append(f'<REF:{ref_idx}>')
            pos += 8
        else:
            return None, offset  # invalid
    return vals, pos

def read_realsxp_at(data, offset, n):
    """Read n doubles from REALSXP at offset."""
    vals = []
    pos = offset + 8  # skip flags + length
    for j in range(n):
        if pos + 8 > len(data):
            break
        v = struct.unpack_from('>d', data, pos)[0]
        vals.append(None if math.isnan(v) else v)
        pos += 8
    return vals, pos

def read_intsxp_at(data, offset, n):
    """Read n integers from INTSXP at offset."""
    vals = []
    pos = offset + 8
    for j in range(n):
        if pos + 4 > len(data):
            break
        v = struct.unpack_from('>i', data, pos)[0]
        vals.append(None if v == -2147483648 else v)
        pos += 4
    return vals, pos

def read_lglsxp_at(data, offset, n):
    """Read n logicals from LGLSXP at offset."""
    vals = []
    pos = offset + 8
    for j in range(n):
        if pos + 4 > len(data):
            break
        v = struct.unpack_from('>i', data, pos)[0]
        vals.append(None if v == -2147483648 else bool(v))
        pos += 4
    return vals, pos

def scan_for_colnames(data, col_names, n_rows):
    """
    Find the location of column names string vector in the binary data,
    then read subsequent column data vectors.
    """
    # First, find the column names block
    # We'll look for STRSXP of length = len(col_names) containing the col names
    n_cols = len(col_names)
    col_bytes = col_names[0].encode('utf-8')
    
    # Search for the first column name as a CHARSXP substring  
    # ORIGPROT encoded in XDR: flags(4) + len(7,4) + "ORIGPROT"
    target = struct.pack('>i', len(col_bytes)) + col_bytes
    
    idx = 0
    while True:
        idx = data.find(target, idx)
        if idx == -1:
            break
        # This might be a CHARSXP for ORIGPROT
        # Check if 8 bytes before is a valid CHARSXP flags word
        if idx >= 8:
            charsxp_flags = struct.unpack_from('>I', data, idx-4)[0]
            sxptype = charsxp_flags & 0xff
            if sxptype == 9:  # CHARSXP
                # Then 4 bytes before that should be the STRSXP flags
                strsxp_flags = struct.unpack_from('>I', data, idx-8-4)[0]
                strsxp_n = struct.unpack_from('>i', data, idx-8)[0] if idx >= 8 else -1
                # Check this works
                strsxp_offset = idx - 8 - 4
                if strsxp_n == n_cols:
                    print(f'Possible col names STRSXP at offset {strsxp_offset}')
                    vals, _ = read_strsxp_at(data, strsxp_offset)
                    if vals and vals[:3] == col_names[:3]:
                        print(f'  Confirmed col names at offset {strsxp_offset}: {vals}')
                        return strsxp_offset
        idx += 1
    return None

def extract_dataframe_from_rda(rda_path, n_rows, col_names_expected):
    """Extract a dataframe from an RDA file by scanning for vectors."""
    
    data = decompress(rda_path)
    print(f'Decompressed: {len(data)} bytes')
    
    n_cols = len(col_names_expected)
    
    # Strategy: Find the VECSXP (type 19) of length n_cols, which is the data.frame body.
    # It will be followed by n_cols vectors each of length n_rows.
    # 
    # Alternative: Find the names STRSXP and then the data vectors.
    
    # First find candidate column name STRSXPs
    col_name_offsets = []
    n = n_cols
    search_offset = 0
    while search_offset < len(data) - 8:
        flags = struct.unpack_from('>I', data, search_offset)[0]
        sxptype = flags & 0xff
        if sxptype == 16:  # STRSXP
            n_check = struct.unpack_from('>i', data, search_offset + 4)[0]
            if n_check == n_cols:
                vals, _ = read_strsxp_at(data, search_offset)
                if vals and col_names_expected[0] in [v for v in vals if v]:
                    col_name_offsets.append((search_offset, vals))
        search_offset += 1
    
    print(f'Found {len(col_name_offsets)} candidate column name blocks')
    for off, vals in col_name_offsets:
        print(f'  Offset {off}: {vals}')
    
    # Now find all vectors of length n_rows
    row_vectors = []
    search_offset = 0
    while search_offset < len(data) - 8:
        flags = struct.unpack_from('>I', data, search_offset)[0]
        sxptype = flags & 0xff
        if sxptype in (14, 13, 16, 10):  # REAL, INT, STR, LGL
            n_check = struct.unpack_from('>i', data, search_offset + 4)[0]
            if n_check == n_rows:
                row_vectors.append((search_offset, sxptype))
        search_offset += 1
    
    print(f'\nFound {len(row_vectors)} vectors of length {n_rows}:')
    for off, sxpt in row_vectors:
        print(f'  offset={off} sxptype={sxpt}')
    
    return row_vectors, col_name_offsets, data


if __name__ == '__main__':
    rda_dir = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2\ADNIMERGE2\build\data'
    
    # ADAS.rda has 12868 rows, 17 cols per docs
    # Cols: ORIGPROT, COLPROT, PTID, RID, VISCODE, VISCODE2, VISDATE, TOTSCORE, TOTAL13,
    #       ID, SITEID, USERDATE, USERDATE2, DD_CRF_VERSION_LABEL, LANGUAGE_CODE, HAS_QC_ERROR, update_stamp
    col_names = ['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE',
                 'TOTSCORE', 'TOTAL13', 'ID', 'SITEID', 'USERDATE', 'USERDATE2',
                 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']
    
    print('=== ADAS.rda ===')
    row_vecs, col_name_offs, data = extract_dataframe_from_rda(
        os.path.join(rda_dir, 'ADAS.rda'), 12868, col_names)
    
    # Now extract each row vector's first few values to identify which is which
    print('\nSampling first 5 values of each row vector:')
    for off, sxpt in row_vecs[:30]:
        try:
            if sxpt == 14:
                vals, _ = read_realsxp_at(data, off, 5)
                print(f'  REAL @{off}: {vals}')
            elif sxpt == 13:
                vals, _ = read_intsxp_at(data, off, 5)
                print(f'  INT  @{off}: {vals}')
            elif sxpt == 16:
                vals, _ = read_strsxp_at(data, off)
                print(f'  STR  @{off}: {vals[:5] if vals else []}')
            elif sxpt == 10:
                vals, _ = read_lglsxp_at(data, off, 5)
                print(f'  LGL  @{off}: {vals}')
        except Exception as e:
            print(f'  @{off} ERROR: {e}')
