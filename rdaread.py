"""Minimal reader for R .rda/.rds (XDR, version 2/3) files -> Python objects.
Supports the types used in airGRdatasets (lists, numeric/int/logical/char vectors, attributes)."""
import lzma, gzip, bz2, struct
import numpy as np

class RObj:
    def __init__(self, value, attrs=None):
        self.value = value
        self.attrs = attrs or {}
    def __repr__(self):
        return f"RObj({type(self.value).__name__}, attrs={list(self.attrs)})"

def _decompress(raw):
    if raw[:2] == b'\x1f\x8b':
        return gzip.decompress(raw)
    if raw[:3] == b'BZh':
        return bz2.decompress(raw)
    if raw[:6] == b'\xfd7zXZ\x00':
        return lzma.decompress(raw)
    return raw

class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0
        self.refs = []
    def i32(self):
        v = struct.unpack('>i', self.d[self.p:self.p+4])[0]; self.p += 4; return v
    def bytes_(self, n):
        b = self.d[self.p:self.p+n]; self.p += n; return b
    def item(self):
        flags = self.i32()
        typ = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if typ == 254:  # NILVALUE
            return None
        if typ == 242:  # EMPTYENV
            return RObj('emptyenv')
        if typ == 253: return RObj('globalenv')
        if typ == 241: return RObj('baseenv')
        if typ == 251: return RObj('missingarg')
        if typ == 252: return RObj('unboundvalue')
        if typ == 255:  # REFSXP
            idx = flags >> 8
            if idx == 0:
                idx = self.i32()
            return self.refs[idx - 1]
        if typ == 1:  # SYMSXP
            s = self.item()
            sym = ('SYM', s)
            self.refs.append(sym)
            return sym
        if typ in (2, 6, 17):  # LISTSXP pairlist / LANGSXP / DOTSXP
            attrs = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car = self.item()
            cdr = self.item()
            node = [(tag[1] if isinstance(tag, tuple) else tag, car)]
            if isinstance(cdr, list):
                node += cdr
            return node
        if typ == 4:  # ENVSXP
            locked = self.i32()
            env = RObj('env'); self.refs.append(env)
            enclos = self.item(); frame = self.item(); hashtab = self.item(); attr = self.item()
            return env
        if typ == 9:  # CHARSXP
            n = self.i32()
            if n == -1:
                return None
            return self.bytes_(n).decode('utf-8', errors='replace')
        if typ in (10, 13):  # LGLSXP, INTSXP
            n = self.i32()
            v = np.frombuffer(self.bytes_(4 * n), dtype='>i4').astype(np.int64)
            return self._finish(v, has_attr)
        if typ == 14:  # REALSXP
            n = self.i32()
            v = np.frombuffer(self.bytes_(8 * n), dtype='>f8').astype(np.float64)
            return self._finish(v, has_attr)
        if typ == 16:  # STRSXP
            n = self.i32()
            v = [self.item() for _ in range(n)]
            return self._finish(v, has_attr)
        if typ in (19, 20):  # VECSXP, EXPRSXP
            n = self.i32()
            v = [self.item() for _ in range(n)]
            return self._finish(v, has_attr)
        raise ValueError(f'unsupported SEXP type {typ} at {self.p}')
    def _finish(self, v, has_attr):
        if has_attr:
            a = self.item()
            attrs = {}
            for k, val in (a or []):
                attrs[k] = val
            return RObj(v, attrs)
        return RObj(v)

def _sym(k):
    return k[1] if isinstance(k, tuple) else k

def read_rda(path):
    raw = _decompress(open(path, 'rb').read())
    assert raw[:5] in (b'RDX2\n', b'RDX3\n'), raw[:5]
    r = Reader(raw[5:])
    assert r.bytes_(2) == b'X\n'
    ver = r.i32(); r.i32(); r.i32()
    if ver == 3:
        n = r.i32(); r.bytes_(n)
    top = r.item()
    return {(_sym(k)): v for k, v in top}

def to_py(o):
    """Convert RObj tree into python dicts/lists/numpy/pandas-friendly structures."""
    if isinstance(o, RObj):
        names = o.attrs.get('names')
        val = o.value
        if isinstance(val, list):
            items = [to_py(x) for x in val]
        else:
            items = val
        cls = o.attrs.get('class')
        cls = cls.value if isinstance(cls, RObj) else None
        if names is not None and isinstance(items, list):
            nm = names.value
            d = {nm[i]: items[i] for i in range(len(nm))}
            if cls and 'data.frame' in cls:
                d['__class__'] = 'data.frame'
            return d
        if cls and 'POSIXct' in cls:
            return np.array(val, dtype='datetime64[s]') if False else ('POSIXct', np.asarray(val))
        return items
    return o
