"""
adni/pipeline.py
=================
End-to-end pipeline:
  1. Discover and load data files.
  2. Build clinical spine from ADAS.rda (TOTAL13 = ADAS13 target).
  3. Merge DXSUM diagnosis labels onto spine.
  4. Attach nearest MRI and FDG-PET scan per visit.
  5. Save merged CSV, column_summary.md, splits.json.
  6. Delegate sequence building to build_sequences.py.
  7. Sanity checks and coverage reporting.

Run from repo root:  python adni/pipeline.py
"""

import os, sys, glob, json, math, struct, bz2, gzip, lzma, warnings, re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent   # c:\Users\tdc65\adni\..\..
ADNI_DIR     = Path(__file__).resolve().parent           # c:\Users\tdc65\adni
RAW_BASE     = ADNI_DIR / "raw_downloads" / "ADNIMERGE2"
RDA_DIR      = RAW_BASE / "ADNIMERGE2" / "build" / "data"
TABLES_DIR   = RAW_BASE / "Tables_24Feb2026"
MRI_IDA_DIR  = RAW_BASE / "MRI_DATA_T1Sagittal3d3test_IDA_Metadata"
PET_IDA_DIR  = RAW_BASE / "Pet18FFDG1f18_IDA_Metadata"
OUT_DIR      = ADNI_DIR / "outputs"
SEQ_DIR      = OUT_DIR / "sequences"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_DIR.mkdir(parents=True, exist_ok=True)

R_ORIGIN = pd.Timestamp("1970-01-01")   # R's date origin

# ──────────────────────────────────────────────────────────────────
# Section 1 – File discovery
# ──────────────────────────────────────────────────────────────────
def discover_files():
    files = {}

    # ADAS.rda (TOTAL13 = ADAS13)
    adas_rda = RDA_DIR / "ADAS.rda"
    if adas_rda.exists():
        files["adas_rda"] = str(adas_rda)
    else:
        print(f"  WARNING: ADAS.rda not found at {adas_rda}")

    # ADSL.rda (demographics)
    adsl_rda = RDA_DIR / "ADSL.rda"
    if adsl_rda.exists():
        files["adsl_rda"] = str(adsl_rda)

    # DXSUM CSV
    dxsum_csv = TABLES_DIR / "All_Subjects_DXSUM_24Feb2026.csv"
    if dxsum_csv.exists():
        files["dxsum_csv"] = str(dxsum_csv)

    # Key_MRI CSV
    key_mri_csv = TABLES_DIR / "All_Subjects_Key_MRI_24Feb2026.csv"
    if key_mri_csv.exists():
        files["key_mri_csv"] = str(key_mri_csv)

    # Key_PET CSV
    key_pet_csv = TABLES_DIR / "All_Subjects_Key_PET_24Feb2026.csv"
    if key_pet_csv.exists():
        files["key_pet_csv"] = str(key_pet_csv)

    print("  Discovered files:")
    for k, v in files.items():
        print(f"    [{k}] {v}")
    return files


# ──────────────────────────────────────────────────────────────────
# Section 2 – ADAS.rda binary decoder
# ──────────────────────────────────────────────────────────────────
def _decompress_rda(path: str) -> bytes:
    with open(path, "rb") as f:
        head = f.read(2)
    if head[:2] == b"BZ":
        with bz2.open(path, "rb") as f:
            return f.read()
    if head[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return f.read()
    try:
        with lzma.open(path, "rb") as f:
            return f.read()
    except Exception:
        with open(path, "rb") as f:
            return f.read()

def _read_strsxp_at(data: bytes, offset: int):
    """Read a STRSXP vector at byte offset; returns (list_of_strings, end_offset)."""
    n = struct.unpack_from(">i", data, offset + 4)[0]
    if not (0 <= n <= 10_000_000):
        return None, offset
    vals = []
    pos = offset + 8
    for _ in range(n):
        if pos + 8 > len(data):
            break
        flags = struct.unpack_from(">I", data, pos)[0]
        sxpt = flags & 0xFF
        if sxpt == 9:  # CHARSXP
            slen = struct.unpack_from(">i", data, pos + 4)[0]
            if slen == -1:
                vals.append(None); pos += 8
            elif 0 <= slen <= 1_000_000:
                s = data[pos + 8 : pos + 8 + slen].decode("utf-8", errors="replace")
                vals.append(s); pos += 8 + slen
            else:
                vals.append(None); pos += 8
        elif sxpt == 254:  # REFSXP
            ref = struct.unpack_from(">i", data, pos + 4)[0]
            vals.append(f"<REF:{ref}>"); pos += 8
        else:
            vals.append(None); pos += 8
    return vals, pos

def _read_realsxp_at(data: bytes, offset: int, n: int):
    """Read n doubles from a REALSXP block (offset → flags word)."""
    vals, pos = [], offset + 8
    na_pattern = struct.pack(">Q", 0x7FF00000000007A2)
    for _ in range(n):
        raw = data[pos : pos + 8]
        v = struct.unpack(">d", raw)[0]
        vals.append(None if math.isnan(v) else v)
        pos += 8
    return vals, pos

def _read_intsxp_at(data: bytes, offset: int, n: int):
    vals, pos = [], offset + 8
    for _ in range(n):
        v = struct.unpack_from(">i", data, pos)[0]
        vals.append(None if v == -2_147_483_648 else v)
        pos += 4
    return vals, pos

def decode_adas_rda(path: str) -> pd.DataFrame:
    """
    Decode ADAS.rda by direct XDR binary scan.
    Returns a DataFrame with columns matching the R data frame.
    Known structure (12 868 rows × 17 cols) from package docs.
    """
    data = _decompress_rda(path)
    n_rows = 12_868
    n_cols = 17

    # Column names confirmed at offset 2 048 035
    col_names = [
        "ORIGPROT","COLPROT","PTID","RID","VISCODE","VISCODE2","VISDATE",
        "TOTSCORE","TOTAL13","ID","SITEID","USERDATE","USERDATE2",
        "DD_CRF_VERSION_LABEL","LANGUAGE_CODE","HAS_QC_ERROR","update_stamp",
    ]

    # Vector byte offsets discovered by scanner:
    #  ORIGPROT  STR   @56
    #  COLPROT   INT   @168244
    #  PTID      STR   @219867
    #  RID       REAL  @452148
    #  VISCODE   STR   @555100
    #  VISCODE2  STR   @696246
    #  VISDATE   REAL  @835525   (R date = days since 1970-01-01)
    #  TOTSCORE  REAL  @938509
    #  TOTAL13   REAL  @1041461  ← ADAS-13 target
    #  ID        REAL  @1144413
    #  SITEID    REAL  @1247365
    #  USERDATE  REAL  @1350317
    #  USERDATE2 REAL  @1453301
    #  D_CRF     STR   @1556285
    #  LANG_CODE STR   @1665477
    #  HAS_QC    STR   @1777165
    #  upd_stamp REAL  @1944973
    vector_info = [
        ("ORIGPROT",            "STR",  56),
        ("COLPROT",             "INT",  168244),
        ("PTID",                "STR",  219867),
        ("RID",                 "REAL", 452148),
        ("VISCODE",             "STR",  555100),
        ("VISCODE2",            "STR",  696246),
        ("VISDATE",             "REAL", 835525),
        ("TOTSCORE",            "REAL", 938509),
        ("TOTAL13",             "REAL", 1041461),
        ("ID",                  "REAL", 1144413),
        ("SITEID",              "REAL", 1247365),
        ("USERDATE",            "REAL", 1350317),
        ("USERDATE2",           "REAL", 1453301),
        ("DD_CRF_VERSION_LABEL","STR",  1556285),
        ("LANGUAGE_CODE",       "STR",  1665477),
        ("HAS_QC_ERROR",        "STR",  1777165),
        ("update_stamp",        "REAL", 1944973),
    ]

    cols = {}
    for col_name, dtype, offset in vector_info:
        try:
            if dtype == "STR":
                vals, _ = _read_strsxp_at(data, offset)
            elif dtype == "REAL":
                vals, _ = _read_realsxp_at(data, offset, n_rows)
            elif dtype == "INT":
                vals, _ = _read_intsxp_at(data, offset, n_rows)
            else:
                vals = [None] * n_rows
            cols[col_name] = vals
        except Exception as exc:
            print(f"  WARNING decoding {col_name}: {exc}")
            cols[col_name] = [None] * n_rows

    df = pd.DataFrame(cols)

    # Convert R date integers → pandas dates
    # R stores dates as floats: days since 1970-01-01
    for date_col in ("VISDATE", "USERDATE", "USERDATE2", "update_stamp"):
        if date_col in df.columns:
            try:
                def _r_to_date(d):
                    if d is None:
                        return pd.NaT
                    try:
                        days = float(d)
                        if not math.isfinite(days) or abs(days) > 50000:
                            return pd.NaT
                        return (R_ORIGIN + timedelta(days=days)).normalize()
                    except (TypeError, ValueError, OverflowError):
                        return pd.NaT
                df[date_col] = pd.to_datetime(df[date_col].apply(_r_to_date), errors="coerce")
                df[date_col] = df[date_col].dt.normalize()
            except Exception as e:
                print(f"  WARNING converting {date_col}: {e}")

    # COLPROT factor → string via levels (ADNI1=1..ADNI4=5)
    colprot_levels = ["ADNI1","ADNIGO","ADNI2","ADNI3","ADNI4"]
    if "COLPROT" in df.columns:
        df["COLPROT"] = df["COLPROT"].apply(
            lambda v: colprot_levels[int(v)-1] if v is not None and 1 <= int(v) <= len(colprot_levels) else None
        )

    # RID to int
    for icol in ("RID","ID","SITEID"):
        if icol in df.columns:
            df[icol] = pd.to_numeric(df[icol], errors="coerce").astype("Int64")

    # TOTAL13 / TOTSCORE numeric
    for ncol in ("TOTAL13","TOTSCORE"):
        if ncol in df.columns:
            df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    return df


def decode_adsl_rda(path: str) -> pd.DataFrame:
    """
    Decode ADSL.rda (demographics: AGE, SEX, EDUC, APOE4) via binary scan.
    Returns a DataFrame or empty DF if extraction fails.
    Known: 157767 bytes compressed. Much smaller than ADAS.
    We scan for vectors using documentated column presence.
    """
    try:
        data = _decompress_rda(path)
    except Exception as e:
        print(f"  WARNING: cannot decompress ADSL.rda: {e}")
        return pd.DataFrame()

    # Scan for STRSXP/REALSXP of the subject count
    # We don't know exact row count for ADSL but scan flexible
    # Strategy: find PTID-like strings to locate the data

    # Search for a STRSXP whose first element looks like NNN_S_NNNN (ADNI PTID)
    ptid_pattern = re.compile(r"^\d{3}_S_\d{4}$")

    # Scan from start for all STRSXPs of reasonable size, check first element
    candidates = []
    i = 0
    while i < min(len(data) - 8, 1_000_000):
        flags = struct.unpack_from(">I", data, i)[0]
        sxpt = flags & 0xFF
        if sxpt == 16:
            n = struct.unpack_from(">i", data, i + 4)[0]
            if 100 <= n <= 50_000:
                try:
                    vals, _ = _read_strsxp_at(data, i)
                    if vals and vals[0] and ptid_pattern.match(str(vals[0])):
                        candidates.append((i, n, vals[:3]))
                except Exception:
                    pass
        i += 1

    if not candidates:
        print("  WARNING: ADSL.rda PTID vector not found – demographics unavailable")
        return pd.DataFrame()

    # Take the first candidate
    ptid_offset, n_rows, _ = candidates[0]
    print(f"  ADSL: found PTID vector at offset {ptid_offset}, n={n_rows}")
    ptid_vals, _ = _read_strsxp_at(data, ptid_offset)

    # Now find numeric columns (RID, AGE) near ptid_offset
    # Scan forward for REALSXP of same length
    results = {"PTID": ptid_vals}
    search_start = ptid_offset + 8
    found_reals = []
    i = search_start
    while i < min(len(data) - 8, ptid_offset + 10_000_000):
        flags = struct.unpack_from(">I", data, i)[0]
        sxpt = flags & 0xFF
        if sxpt == 14:
            n = struct.unpack_from(">i", data, i + 4)[0]
            if n == n_rows:
                vals, _ = _read_realsxp_at(data, i, n_rows)
                found_reals.append((i, vals[:5]))
        i += 1

    # Heuristic: AGE is typically 50-100, EDUC is 0-25, APOE4 is 0/1/2
    for off, sample in found_reals:
        non_null = [v for v in sample if v is not None]
        if not non_null:
            continue
        mn = min(non_null); mx = max(non_null)
        if 50 <= mn and mx <= 105:
            if "AGE" not in results:
                results["AGE"], _ = _read_realsxp_at(data, off, n_rows)
        elif 0 <= mn and mx <= 30:
            if "EDUC" not in results:
                results["EDUC"], _ = _read_realsxp_at(data, off, n_rows)

    # Also look for APOE4 (integers 0/1/2) in INT vectors
    i = search_start
    while i < min(len(data) - 8, ptid_offset + 10_000_000):
        flags = struct.unpack_from(">I", data, i)[0]
        sxpt = flags & 0xFF
        if sxpt == 13:
            n = struct.unpack_from(">i", data, i + 4)[0]
            if n == n_rows:
                vals, _ = _read_intsxp_at(data, i, n_rows)
                non_null = [v for v in vals[:20] if v is not None]
                if non_null and all(v in (0, 1, 2) for v in non_null) and "APOE4" not in results:
                    results["APOE4"] = vals
        i += 1

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────
# Section 2b – Build clinical spine from ADAS.rda + optional ADSL
# ──────────────────────────────────────────────────────────────────
def build_clinical_spine(files: dict) -> pd.DataFrame:
    print("\n[Step 2] Building clinical spine from ADAS.rda ...")

    # Load ADAS
    adas = decode_adas_rda(files["adas_rda"])
    print(f"  ADAS decoded: {adas.shape}")

    # Rename TOTAL13 → ADAS13 and VISDATE → EXAMDATE
    spine = adas.rename(columns={"TOTAL13": "ADAS13", "VISDATE": "EXAMDATE"}).copy()

    # Keep only rows where ADAS13 is not null
    spine = spine.dropna(subset=["ADAS13"])
    print(f"  After dropping rows with ADAS13=NaN: {spine.shape}")

    # Keep core columns
    keep = ["PTID","RID","VISCODE","VISCODE2","EXAMDATE","ORIGPROT","COLPROT","ADAS13","TOTSCORE"]
    spine = spine[[c for c in keep if c in spine.columns]].copy()

    # Ensure EXAMDATE is datetime
    if "EXAMDATE" in spine.columns:
        spine["EXAMDATE"] = pd.to_datetime(spine["EXAMDATE"], errors="coerce")

    # Merge demographics if available
    if "adsl_rda" in files:
        try:
            adsl = decode_adsl_rda(files["adsl_rda"])
            if not adsl.empty and "PTID" in adsl.columns:
                dem_cols = ["PTID"] + [c for c in ["AGE","EDUC","APOE4","SEX"] if c in adsl.columns]
                adsl_sub = adsl[dem_cols].drop_duplicates("PTID")
                spine = spine.merge(adsl_sub, on="PTID", how="left")
                print(f"  Merged demographics: {dem_cols}")
        except Exception as exc:
            print(f"  WARNING: demographics merge failed: {exc}")

    print(f"  Clinical spine: {spine.shape[0]} rows, {spine.shape[1]} cols")
    return spine


# ──────────────────────────────────────────────────────────────────
# Section 3 – Merge DXSUM
# ──────────────────────────────────────────────────────────────────
def merge_dxsum(spine: pd.DataFrame, files: dict) -> pd.DataFrame:
    print("\n[Step 3] Merging DXSUM ...")
    if "dxsum_csv" not in files:
        print("  WARNING: dxsum_csv not found – skipping diagnosis merge")
        return spine

    dx = pd.read_csv(files["dxsum_csv"], low_memory=False)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    print(f"  DXSUM loaded: {dx.shape}")

    dx_keep = ["RID","PTID","VISCODE","VISCODE2","EXAMDATE","DIAGNOSIS","DXNORM"]
    dx_sub = dx[[c for c in dx_keep if c in dx.columns]].copy()
    dx_sub = dx_sub.drop_duplicates(subset=[c for c in ["RID","VISCODE"] if c in dx_sub.columns])

    # Strategy 1: join on RID + VISCODE2
    merged = spine.copy()
    if "RID" in merged.columns and "VISCODE2" in merged.columns and "RID" in dx_sub.columns and "VISCODE2" in dx_sub.columns:
        dx_1 = dx_sub.rename(columns={"DIAGNOSIS":"dx_DIAGNOSIS","DXNORM":"dx_DXNORM"})
        dx_1 = dx_1[["RID","VISCODE2"] + [c for c in ["dx_DIAGNOSIS","dx_DXNORM"] if c in dx_1.columns]]
        merged = merged.merge(dx_1, on=["RID","VISCODE2"], how="left")

    # Strategy 2: fallback nearest-date match (within ±30 days) for unmatched rows
    if "dx_DIAGNOSIS" in merged.columns:
        unmatched_mask = merged["dx_DIAGNOSIS"].isna() & merged["EXAMDATE"].notna()
        n_unmatched = unmatched_mask.sum()
        if n_unmatched > 0 and "EXAMDATE" in dx_sub.columns:
            print(f"  Fallback nearest-date matching for {n_unmatched} unmatched rows ...")
            dx_date = dx_sub.copy()
            dx_date["EXAMDATE"] = pd.to_datetime(dx_date["EXAMDATE"], errors="coerce")
            # For each unmatched row, find nearest DX row for same subject
            result_diag = merged.loc[unmatched_mask, "dx_DIAGNOSIS"].copy()
            result_norm  = merged.loc[unmatched_mask, "dx_DXNORM"].copy()
            for idx in merged.index[unmatched_mask]:
                ptid = merged.at[idx, "PTID"]
                edate = merged.at[idx, "EXAMDATE"]
                if pd.isna(edate):
                    continue
                subj_dx = dx_date[dx_date["PTID"] == ptid] if "PTID" in dx_date.columns else pd.DataFrame()
                if subj_dx.empty:
                    continue
                subj_dx = subj_dx.copy()
                subj_dx["_delta"] = (subj_dx["EXAMDATE"] - edate).abs()
                nearest = subj_dx.sort_values("_delta").iloc[0]
                if nearest["_delta"] <= pd.Timedelta(days=30):
                    result_diag[idx] = nearest.get("DIAGNOSIS", None)
                    result_norm[idx]  = nearest.get("DXNORM", None)
            merged.loc[unmatched_mask, "dx_DIAGNOSIS"] = result_diag
            merged.loc[unmatched_mask, "dx_DXNORM"]    = result_norm

    # Finalise column names
    for old, new in [("dx_DIAGNOSIS","DIAGNOSIS"),("dx_DXNORM","DXNORM")]:
        if old in merged.columns:
            merged.rename(columns={old: new}, inplace=True)

    matched = merged["DIAGNOSIS"].notna().sum() if "DIAGNOSIS" in merged.columns else 0
    print(f"  DXSUM merge: {matched}/{len(merged)} rows with diagnosis")
    return merged


# ──────────────────────────────────────────────────────────────────
# Section 4 – Imaging metadata and nearest-scan matching
# ──────────────────────────────────────────────────────────────────
FDG_KEYWORDS  = {"fdg","18f-fdg","18 f fdg","fluorodeoxyglucose"}
TAU_KEYWORDS  = {"av1451","flortaucipir","tau","mk6240","mk-6240","pi2620","pi-2620","nav4694"}

def _is_fdg(val: str) -> bool:
    if not val:
        return False
    v = val.lower().replace("-","").replace(" ","")
    return any(k.replace("-","").replace(" ","") in v for k in FDG_KEYWORDS)

def _is_tau(val: str) -> bool:
    if not val:
        return False
    v = val.lower().replace("-","").replace(" ","")
    return any(k.replace("-","").replace(" ","") in v for k in TAU_KEYWORDS)


def build_mri_meta(files: dict) -> pd.DataFrame:
    """Build per-image MRI metadata table."""
    print("\n[Step 4a] Building MRI metadata ...")
    frames = []

    if "key_mri_csv" in files:
        df = pd.read_csv(files["key_mri_csv"], low_memory=False)
        df["acq_date"] = pd.to_datetime(df.get("image_date"), errors="coerce")
        df["ptid"]        = df.get("subject_id")
        df["modality"]    = "MRI"
        df["description"] = df.get("series_description")
        df["image_id"]    = df.get("image_id")
        sub = df[["ptid","acq_date","modality","description","image_id"]].copy()
        frames.append(sub)
        print(f"  Key_MRI: {len(df)} rows")

    if not frames:
        print("  WARNING: No MRI metadata found")
        return pd.DataFrame(columns=["ptid","acq_date","modality","description","image_id"])

    mri_meta = pd.concat(frames, ignore_index=True)
    mri_meta = mri_meta.dropna(subset=["ptid","acq_date"])
    print(f"  MRI metadata total: {mri_meta.shape[0]} rows")
    return mri_meta


def build_pet_meta(files: dict) -> pd.DataFrame:
    """Build per-image PET metadata (FDG only)."""
    print("\n[Step 4b] Building PET metadata (FDG filter) ...")
    frames = []

    if "key_pet_csv" in files:
        df = pd.read_csv(files["key_pet_csv"], low_memory=False)
        df["acq_date"] = pd.to_datetime(df.get("image_date"), errors="coerce")
        df["ptid"]        = df.get("subject_id")
        df["modality"]    = "PET"
        df["description"] = df.get("pet_description")
        df["image_id"]    = df.get("image_id")
        df["tracer"]      = df.get("radiopharmaceutical","")

        # FDG filter: keep if tracer col says 18F-FDG, OR description has 'FDG'
        # and does NOT have tau keywords
        def keep_pet_row(row):
            tracer = str(row.get("tracer","") or "")
            desc   = str(row.get("description","") or "")
            if _is_tau(tracer) or _is_tau(desc):
                return False
            if _is_fdg(tracer) or _is_fdg(desc):
                return True
            # If tau_pet column is 'y'/'yes'/'1' skip it
            tau_flag = str(row.get("tau_pet","") or "").lower()
            if tau_flag in ("y","yes","1","true"):
                return False
            return False  # exclude ambiguous

        mask = df.apply(keep_pet_row, axis=1)
        df_fdg = df[mask].copy()
        sub = df_fdg[["ptid","acq_date","modality","description","image_id","tracer"]].copy()
        frames.append(sub)
        print(f"  Key_PET total: {len(df)}, FDG-only: {len(df_fdg)}")

        # Print unique tracers / descriptions after filtering
        print("  Top 20 unique descriptions after FDG filter:")
        for d in df_fdg["description"].value_counts().head(20).index:
            print(f"    {d}")

    if not frames:
        print("  WARNING: No PET metadata found")
        return pd.DataFrame(columns=["ptid","acq_date","modality","description","image_id","tracer"])

    pet_meta = pd.concat(frames, ignore_index=True)
    pet_meta = pet_meta.dropna(subset=["ptid","acq_date"])
    print(f"  FDG-PET metadata total: {pet_meta.shape[0]} rows")
    return pet_meta


def attach_nearest_scan(spine: pd.DataFrame,
                        img_meta: pd.DataFrame,
                        prefix: str,
                        window_days: int = 90) -> pd.DataFrame:
    """
    For each row in spine, find the nearest scan in img_meta for the same subject
    within ±window_days. Attaches {prefix}_acq_date, {prefix}_image_id,
    {prefix}_description, has_{prefix} columns.
    """
    out = spine.copy()
    acq_col  = f"{prefix}_acq_date"
    iid_col  = f"{prefix}_image_id"
    desc_col = f"{prefix}_description"
    flag_col = f"has_{prefix}"

    out[acq_col]  = pd.NaT
    out[iid_col]  = pd.NA
    out[desc_col] = pd.NA
    out[flag_col] = False

    if img_meta.empty or "ptid" not in img_meta.columns:
        return out

    img_meta = img_meta.copy()
    # Normalise ptid to plain str (avoids ArrowDtype mismatch)
    img_meta["ptid"] = img_meta["ptid"].astype(str)
    img_meta["acq_date"] = pd.to_datetime(img_meta["acq_date"], errors="coerce").dt.normalize()
    img_meta = img_meta.dropna(subset=["acq_date"])

    # Normalise EXAMDATE to date-only
    if "EXAMDATE" in out.columns:
        out["EXAMDATE"] = pd.to_datetime(out["EXAMDATE"], errors="coerce").dt.normalize()

    # Group by ptid for faster lookup
    img_by_subj = img_meta.groupby("ptid")

    for idx, row in out.iterrows():
        ptid  = str(row.get("PTID") or "")
        edate = row.get("EXAMDATE")
        if not ptid or ptid in ("None", "nan") or pd.isna(edate):
            continue
        if ptid not in img_by_subj.groups:
            continue
        subj_scans = img_by_subj.get_group(ptid)
        try:
            deltas = (subj_scans["acq_date"] - edate).abs()
        except Exception:
            continue
        best_idx = deltas.idxmin()
        best_delta = deltas[best_idx]
        if best_delta <= pd.Timedelta(days=window_days):
            best = subj_scans.loc[best_idx]
            out.at[idx, acq_col]  = best["acq_date"]
            out.at[idx, iid_col]  = best.get("image_id", pd.NA)
            out.at[idx, desc_col] = best.get("description", pd.NA)
            out.at[idx, flag_col] = True

    n_matched = out[flag_col].sum()
    print(f"  Attached {prefix}: {n_matched}/{len(out)} visits matched within ±{window_days} days")
    return out


# ──────────────────────────────────────────────────────────────────
# Section 5 – Save outputs
# ──────────────────────────────────────────────────────────────────
def save_merged_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"\n[Step 5a] Saved merged CSV: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def save_column_summary(df: pd.DataFrame, path: str):
    rows = []
    for col in df.columns:
        miss = df[col].isna().sum()
        miss_pct = 100.0 * miss / max(len(df), 1)
        rows.append({
            "column":   col,
            "dtype":    str(df[col].dtype),
            "n_notnull":int(len(df) - miss),
            "missing_n":int(miss),
            "missing_%":f"{miss_pct:.1f}",
        })
    summary_df = pd.DataFrame(rows)
    # Write markdown
    lines = ["# ADNI Spine ADAS13 Multimodal – Column Summary\n"]
    lines.append(f"**Rows:** {len(df)}   **Columns:** {len(df.columns)}\n\n")
    lines.append("| column | dtype | n_notnull | missing_n | missing_% |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, r in summary_df.iterrows():
        lines.append(f"| {r['column']} | {r['dtype']} | {r['n_notnull']} | {r['missing_n']} | {r['missing_%']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Step 5b] Saved column summary: {path}")


def save_splits(df: pd.DataFrame, path: str, seed: int = 42):
    """Split subjects 70/15/15 with no leakage across rows."""
    subj_col = "PTID" if "PTID" in df.columns else "RID"
    subjects = df[subj_col].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    splits = {
        "train": subjects[:n_train],
        "val":   subjects[n_train:n_train+n_val],
        "test":  subjects[n_train+n_val:],
    }
    splits_serialisable = {k: [str(s) for s in v] for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits_serialisable, f, indent=2)
    print(f"[Step 5c] Saved splits: {path}  (train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} subjects)")
    return splits


# ──────────────────────────────────────────────────────────────────
# Section 7 – Sanity checks
# ──────────────────────────────────────────────────────────────────
def run_sanity_checks(spine: pd.DataFrame, mri_meta: pd.DataFrame, pet_meta: pd.DataFrame):
    print("\n" + "="*60)
    print("[Step 7] SANITY CHECKS & REPORTING")
    print("="*60)

    subj_col = "PTID" if "PTID" in spine.columns else "RID"
    n_subjects = spine[subj_col].nunique() if subj_col in spine.columns else 0
    n_rows     = len(spine)
    print(f"  Subjects:   {n_subjects}")
    print(f"  Visit rows: {n_rows}")

    if subj_col in spine.columns:
        visits_per_subj = spine.groupby(subj_col).size()
        print(f"  Visits/subject – min:{visits_per_subj.min()}  median:{visits_per_subj.median():.1f}  max:{visits_per_subj.max()}")

    if "ADAS13" in spine.columns:
        miss_adas = spine["ADAS13"].isna().sum()
        print(f"  ADAS13 missing: {miss_adas}/{n_rows} ({100*miss_adas/max(n_rows,1):.1f}%)")
        print(f"  ADAS13 range:   {spine['ADAS13'].min():.1f}–{spine['ADAS13'].max():.1f},  mean={spine['ADAS13'].mean():.2f}")

    print("\n  Imaging coverage:")
    for flag_col, label in [("has_mri","MRI"),("has_fdg_pet","FDG-PET")]:
        if flag_col in spine.columns:
            n_match = spine[flag_col].sum()
            pct = 100*n_match / max(n_rows,1)
            print(f"    {label}: {n_match}/{n_rows} ({pct:.1f}%) visits matched")

            # Typical date diff
            date_col = f"{'mri' if label=='MRI' else 'fdg_pet'}_acq_date"
            if date_col in spine.columns and "EXAMDATE" in spine.columns:
                matched_rows = spine[spine[flag_col] == True]
                if not matched_rows.empty:
                    deltas = (matched_rows[date_col] - matched_rows["EXAMDATE"]).dt.days.abs()
                    print(f"    {label} date diff – median:{deltas.median():.0f}d  mean:{deltas.mean():.0f}d")

    print("\n  Top 20 unique FDG-PET descriptions after filtering:")
    if not pet_meta.empty and "description" in pet_meta.columns:
        for d in pet_meta["description"].value_counts().head(20).index:
            print(f"    {repr(d)}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ADNI Multimodal Pipeline")
    print("=" * 60)

    # 1. Discover files
    print("\n[Step 1] Discovering data files ...")
    files = discover_files()

    # 2. Clinical spine
    spine = build_clinical_spine(files)

    # 3. Merge DXSUM
    spine = merge_dxsum(spine, files)

    # 4. Imaging metadata
    mri_meta = build_mri_meta(files)
    pet_meta = build_pet_meta(files)

    # Attach nearest scans
    spine = attach_nearest_scan(spine, mri_meta, prefix="mri", window_days=90)
    spine = attach_nearest_scan(spine, pet_meta, prefix="fdg_pet", window_days=90)

    # 5. Save outputs
    csv_path    = OUT_DIR / "adni_spine_adas13_multimodal.csv"
    md_path     = OUT_DIR / "column_summary.md"
    splits_path = OUT_DIR / "splits.json"

    save_merged_csv(spine, str(csv_path))
    save_column_summary(spine, str(md_path))
    splits = save_splits(spine, str(splits_path))

    # 6. Build sequences (delegate to build_sequences.py)
    print("\n[Step 6] Building Neural CDE sequences ...")
    seq_script = Path(__file__).parent / "build_sequences.py"
    if seq_script.exists():
        import subprocess
        ret = subprocess.run([sys.executable, str(seq_script),
                              "--csv", str(csv_path),
                              "--out", str(SEQ_DIR),
                              "--splits", str(splits_path)],
                             capture_output=False)
        if ret.returncode != 0:
            print(f"  WARNING: build_sequences.py exited with code {ret.returncode}")
    else:
        print(f"  WARNING: {seq_script} not found – skipping sequence build")

    # 7. Sanity checks
    run_sanity_checks(spine, mri_meta, pet_meta)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"  Merged CSV:   {csv_path}")
    print(f"  Col summary:  {md_path}")
    print(f"  Splits:       {splits_path}")
    print(f"  Sequences:    {SEQ_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
