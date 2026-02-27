"""
Parse R XDR v3 RDA files with ALTREP support.
ALTREP (Alternative Representation) was introduced in R 3.5.
For compact integer/real sequences (ALTREP), we derive the data from
the state object (a VECSXP with [start, end] or similar).
"""
import bz2, gzip, lzma, struct, os, math, re, io

class RDAParseError(Exception):
    pass

class RDAParser:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.ref_table = [None]  # 1-indexed

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        result = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return result[0] if len(result)==1 else result

    def read_int(self):
        return self.unpack('>i')

    def read_double(self):
        return self.unpack('>d')

    def read_bytes(self, n):
        result = self.data[self.pos:self.pos+n]
        self.pos += n
        return result

    def read_string_raw(self):
        """Read a CHARSXP-like raw string: 4-byte length then bytes."""
        length = self.read_int()
        if length == -1:
            return None  # NA_character_
        return self.read_bytes(length).decode('utf-8', errors='replace')

    def read_flags(self):
        val = self.read_int()
        sxptype = val & 0xff
        levels = (val >> 12) & 0xfffff
        is_obj = (val >> 8) & 1
        has_attr = (val >> 9) & 1
        has_tag = (val >> 10) & 1
        return sxptype, levels, is_obj, has_attr, has_tag

    def read_object(self):
        if self.pos >= len(self.data):
            return None

        sxptype, levels, is_obj, has_attr, has_tag = self.read_flags()

        # Special packed types (>= 240)
        if sxptype == 254:  # REFSXP
            ref_idx = self.read_int()
            return self.ref_table[ref_idx] if ref_idx < len(self.ref_table) else f'<REF:{ref_idx}>'

        if sxptype == 251:  return None  # NILVALUE
        if sxptype == 255:  return '<GLOBALENV>'
        if sxptype == 253:  return '<EMPTYENV>'
        if sxptype == 252:  return '<BASEENV>'
        if sxptype == 239:  return '<BASENAMESPACE>'

        # Most sxptypes go into the reference table
        ref_pos = len(self.ref_table)
        self.ref_table.append(None)  # placeholder

        attr = None
        tag = None
        if has_attr:
            attr = self.read_object()
        if has_tag:
            tag = self.read_object()

        result = None

        if sxptype == 0:  # NILSXP
            result = None

        elif sxptype == 1:  # SYMSXP
            name = self.read_object()
            result = f'<SYM:{name}>'

        elif sxptype == 2:  # LISTSXP (pairlist)
            car = self.read_object()
            cdr = self.read_object()
            result = {'type': 'pairlist', 'tag': tag, 'car': car, 'cdr': cdr, 'attr': attr}

        elif sxptype == 4:  # ENVSXP
            locked = self.read_int()
            enclos = self.read_object()
            frame = self.read_object()
            hashtab = self.read_object()
            result = {'type': 'env', 'frame': frame}

        elif sxptype == 3:  # CLOSXP (closure / function)
            formals = self.read_object()
            body = self.read_object()
            env = self.read_object()
            result = {'type': 'closure', 'body': body}

        elif sxptype == 5:  # PROMSXP
            value = self.read_object()
            expr = self.read_object()
            env = self.read_object()
            result = {'type': 'promise', 'expr': expr}

        elif sxptype == 6:  # LANGSXP
            car = self.read_object()
            cdr = self.read_object()
            result = {'type': 'lang', 'car': car, 'cdr': cdr}

        elif sxptype == 9:  # CHARSXP
            s = self.read_string_raw()
            result = s

        elif sxptype == 10:  # LGLSXP
            n = self.read_int()
            vals = []
            for _ in range(n):
                v = self.read_int()
                vals.append(None if v == -2147483648 else bool(v))
            result = {'type': 'logical', 'data': vals, 'attr': attr}

        elif sxptype == 13:  # INTSXP
            n = self.read_int()
            vals = []
            for _ in range(n):
                v = self.read_int()
                vals.append(None if v == -2147483648 else v)
            result = {'type': 'integer', 'data': vals, 'attr': attr}

        elif sxptype == 14:  # REALSXP
            n = self.read_int()
            vals = []
            for _ in range(n):
                v = self.read_double()
                nan_payload = struct.unpack('>Q', struct.pack('>d', v))[0] if not math.isnan(v) else 0
                vals.append(None if math.isnan(v) else v)
            result = {'type': 'real', 'data': vals, 'attr': attr}

        elif sxptype == 16:  # STRSXP
            n = self.read_int()
            vals = []
            for _ in range(n):
                v = self.read_object()
                vals.append(v)
            result = {'type': 'character', 'data': vals, 'attr': attr}

        elif sxptype == 17:  # DOTSXP
            car = self.read_object()
            cdr = self.read_object()
            result = {'type': 'dots'}

        elif sxptype == 19:  # VECSXP (list)
            n = self.read_int()
            vals = []
            for _ in range(n):
                v = self.read_object()
                vals.append(v)
            result = {'type': 'list', 'data': vals, 'attr': attr}

        elif sxptype == 20:  # EXPRSXP
            n = self.read_int()
            vals = [self.read_object() for _ in range(n)]
            result = {'type': 'expression', 'data': vals}

        elif sxptype == 24:  # RAWSXP
            n = self.read_int()
            result = {'type': 'raw', 'data': self.read_bytes(n)}

        elif sxptype == 241:  # PACKAGESXP
            s = self.read_object()
            result = f'<PKG:{s}>'

        elif sxptype == 242:  # NAMESPACESXP  
            v = self.read_object()
            result = '<NAMESPACE>'

        elif sxptype == 248:  # ATTRLANGSXP
            # Attributed LANGSXP
            obj = self.read_object()
            result = obj

        elif sxptype == 249:  # ATTRLISTSXP
            obj = self.read_object()
            result = obj

        elif sxptype == 250:  # ALTREP_SXP
            # Alternative representation - key for compact sequences
            info = self.read_object()   # class info (LISTSXP with class info)
            state = self.read_object()  # state (depends on class)
            attr2 = self.read_object()  # attributes
            # Try to expand the ALTREP
            result = self._expand_altrep(info, state, attr2, attr)

        elif sxptype == 15:  # CPLXSXP
            n = self.read_int()
            for _ in range(n):
                self.read_double()
                self.read_double()
            result = {'type': 'complex', 'len': n}

        else:
            # Unknown type - problematic
            raise RDAParseError(f'Unknown sxptype {sxptype} at offset {self.pos - 4}')

        # Update reference table
        self.ref_table[ref_pos] = result
        return result

    def _expand_altrep(self, info, state, attr2, attr):
        """Try to expand ALTREP objects."""
        # info is typically a pairlist: car=class_name_sym, cdr=pairlist(package_sym)
        # Common ALTREP classes: compact_intseq, compact_realseq, deferred_string, etc.
        
        class_name = None
        if isinstance(info, dict) and info.get('type') == 'pairlist':
            car = info.get('car', '')
            if isinstance(car, str):
                class_name = car.replace('<SYM:', '').rstrip('>')
        
        if class_name and 'intseq' in class_name.lower():
            # compact_intseq: state is REALSXP [length, start, step]
            if isinstance(state, dict) and state.get('type') == 'real':
                d = state['data']
                if len(d) >= 3:
                    length, start, step = int(d[0]), int(d[1]), int(d[2])
                    return {'type': 'integer', 'data': list(range(start, start + length*step, step)), 'attr': attr}
        elif class_name and 'realseq' in class_name.lower():
            # compact_realseq: state is REALSXP [length, start, step]
            if isinstance(state, dict) and state.get('type') == 'real':
                d = state['data']
                if len(d) >= 3:
                    length, start, step = int(d[0]), d[1], d[2]
                    return {'type': 'real', 'data': [start + i*step for i in range(length)], 'attr': attr}
        
        # For other ALTREP types, return placeholder
        return {'type': f'altrep_{class_name}', 'state': state, 'attr': attr}


def load_rda(path):
    """Load an R .rda file and return a dict of name->object."""
    # Decompress
    with open(path, 'rb') as f:
        head = f.read(3)
    if head[:2] == b'BZ':
        with bz2.open(path, 'rb') as f:
            data = f.read()
    elif head[:2] == b'\x1f\x8b':
        with gzip.open(path, 'rb') as f:
            data = f.read()
    elif head[:3] == b'\xfd7z':
        import lzma
        with lzma.open(path, 'rb') as f:
            data = f.read()
    else:
        with open(path, 'rb') as f:
            data = f.read()

    # Parse header
    magic = data[:5]
    if magic == b'RDX3\n':
        fmt_char = chr(data[5])
        pos = 7  # "RDX3\nX\n"
    elif magic[:4] == b'RDA2':
        fmt_char = chr(data[5])
        pos = 7
    else:
        raise RDAParseError(f'Unknown magic: {magic}')

    parser = RDAParser(data)
    parser.pos = pos

    # Read 3 version ints
    v1 = parser.read_int()
    v2 = parser.read_int()
    v3 = parser.read_int()

    # Read root object (usually a pairlist or list of named objects)
    root = parser.read_object()
    return root


def extract_attrs(attr):
    """Extract a dict from a pairlist attribute chain."""
    result = {}
    node = attr
    while node is not None and isinstance(node, dict):
        t = node.get('type', '')
        if t == 'pairlist':
            tag = node.get('tag', '')
            if isinstance(tag, str):
                tag = tag.replace('<SYM:', '').rstrip('>')
            car = node.get('car')
            result[tag] = car
            node = node.get('cdr')
        else:
            break
    return result


def rda_to_dataframe(root):
    """Convert parsed R object to pandas DataFrame."""
    import pandas as pd

    # Root is typically a pairlist of name->value pairs
    # where each value is a data.frame
    dfs = {}
    node = root
    while node is not None and isinstance(node, dict):
        t = node.get('type', '')
        if t == 'pairlist':
            tag = node.get('tag', '')
            if isinstance(tag, str):
                name = tag.replace('<SYM:', '').rstrip('>')
            else:
                name = str(tag)
            val = node.get('car')
            df = try_make_df(val)
            if df is not None:
                dfs[name] = df
            node = node.get('cdr')
        else:
            break
    return dfs


def try_make_df(obj):
    """Try to convert an R object to a pandas DataFrame."""
    import pandas as pd
    if not isinstance(obj, dict):
        return None

    typ = obj.get('type', '')
    attr = obj.get('attr')
    attrs = extract_attrs(attr) if attr else {}

    # A data.frame is a list (VECSXP) with class="data.frame"
    # and attributes: names, row.names, class
    class_attr = attrs.get('class')
    names_attr = attrs.get('names')
    
    # Check class
    is_df = False
    if isinstance(class_attr, dict) and class_attr.get('type') == 'character':
        if 'data.frame' in (class_attr.get('data') or []):
            is_df = True
    
    col_names = None
    if isinstance(names_attr, dict) and names_attr.get('type') == 'character':
        col_names = names_attr.get('data', [])
    
    if typ == 'list' and col_names:
        cols = {}
        data_items = obj.get('data', [])
        for i, col_name in enumerate(col_names):
            if i < len(data_items):
                item = data_items[i]
                if isinstance(item, dict):
                    item_data = item.get('data', [])
                    item_attr = item.get('attr')
                    item_attrs = extract_attrs(item_attr) if item_attr else {}
                    # Check for factor levels
                    levels_attr = item_attrs.get('levels')
                    class2 = item_attrs.get('class')
                    is_factor = False
                    if isinstance(class2, dict) and 'factor' in (class2.get('data') or []):
                        is_factor = True
                    levels = None
                    if isinstance(levels_attr, dict) and levels_attr.get('type') == 'character':
                        levels = levels_attr.get('data', [])
                    if is_factor and levels and isinstance(item_data, list):
                        cols[col_name] = [levels[v-1] if v is not None and 1 <= v <= len(levels) else None for v in item_data]
                    elif isinstance(item_data, list):
                        cols[col_name] = item_data
                    elif isinstance(item, (list, str)):
                        cols[col_name] = item
                elif isinstance(item, list):
                    cols[col_name] = item
                else:
                    cols[col_name] = item
        try:
            return pd.DataFrame(cols)
        except Exception as e:
            return pd.DataFrame({k: v if isinstance(v, list) else [v] for k, v in cols.items()})
    
    return None


if __name__ == '__main__':
    rda_dir = r'c:\Users\tdc65\adni\raw_downloads\ADNIMERGE2\ADNIMERGE2\build\data'
    
    # Try ADAS.rda
    print('=== ADAS.rda ===')
    try:
        root = load_rda(os.path.join(rda_dir, 'ADAS.rda'))
        print('Root type:', type(root), 'dict type:', root.get('type') if isinstance(root, dict) else 'N/A')
        dfs = rda_to_dataframe(root)
        print('DataFrames found:', list(dfs.keys()))
        for name, df in dfs.items():
            print(f'{name}: {df.shape}')
            print('Cols:', list(df.columns)[:20])
    except Exception as e:
        import traceback
        traceback.print_exc()
