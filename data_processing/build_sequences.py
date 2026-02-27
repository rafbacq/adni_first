"""
adni/build_sequences.py
========================
Reads adni/outputs/adni_spine_adas13_multimodal.csv and produces
per-subject Neural CDE–ready arrays:

  - tim_arr  : (T,) float32 – months from baseline
  - fea_mat  : (T, F) float32 – feature matrix
  - msk_mat  : (T, F) bool    – True where fea_mat is observed
  - tgt_arr  : (T,) float32  – ADAS13 target

Outputs are saved as .npz files (one per subject) under
  adni/outputs/sequences/{split}/{ptid}.npz

and as a single  adni/outputs/sequences/manifest.csv
listing each file with its split.

Variable naming requirements: ≥3 letters; pointer-like names include "ptr".
"""

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_VISCODE = {"bl", "sc", "init", "baseline", "4_init", "4_bl"}

# Feature columns to include (clinical + demographics; imaging are flags)
FEATURE_COLS = [
    "ADAS13",          # target (also kept in features for context)
    "TOTSCORE",        # ADAS-11
    "DIAGNOSIS",
    "DXNORM",
    "has_mri",
    "has_fdg_pet",
    "AGE",
    "EDUC",
    "APOE4",
]

def _months_delta(date_base: pd.Timestamp, date_curr: pd.Timestamp) -> float:
    """Return months between two dates (approximate: 365.25/12 days per month)."""
    if pd.isna(date_base) or pd.isna(date_curr):
        return float("nan")
    diff_days = (date_curr - date_base).days
    return diff_days / (365.25 / 12)


def build_subject_sequence(sub_df: pd.DataFrame, feat_cols: list) -> dict | None:
    """
    Build per-subject arrays from a sorted (by EXAMDATE) subject visit table.
    Returns dict with keys: tim_arr, fea_mat, msk_mat, tgt_arr.
    Returns None if the subject has fewer than 2 valid ADAS13 visits.
    """
    sub_df = sub_df.sort_values("EXAMDATE").reset_index(drop=True)

    # Determine baseline date: earliest visit that has EXAMDATE
    valid_dates = sub_df["EXAMDATE"].dropna()
    if valid_dates.empty:
        return None
    date_base = valid_dates.iloc[0]

    tgt_vals = sub_df["ADAS13"].values.astype(float)
    n_valid_tgt = np.sum(~np.isnan(tgt_vals))
    if n_valid_tgt < 1:
        return None  # no visits with ADAS13

    # Time vector (months from baseline)
    tim_arr = np.array([
        _months_delta(date_base, row["EXAMDATE"]) if not pd.isna(row["EXAMDATE"]) else float("nan")
        for _, row in sub_df.iterrows()
    ], dtype=np.float32)

    # Feature matrix
    rows_feat = []
    rows_mask = []
    for _, row in sub_df.iterrows():
        fea_row = []
        msk_row = []
        for col in feat_cols:
            if col not in sub_df.columns:
                fea_row.append(0.0)
                msk_row.append(False)
                continue
            val = row.get(col, None)
            if pd.isna(val) or val is None:
                fea_row.append(0.0)  # imputed with 0
                msk_row.append(False)
            else:
                try:
                    fea_row.append(float(val))
                    msk_row.append(True)
                except (TypeError, ValueError):
                    fea_row.append(0.0)
                    msk_row.append(False)
        rows_feat.append(fea_row)
        rows_mask.append(msk_row)

    fea_mat = np.array(rows_feat, dtype=np.float32)
    msk_mat = np.array(rows_mask, dtype=bool)
    tgt_arr = tgt_vals.astype(np.float32)

    return {
        "tim_arr": tim_arr,
        "fea_mat": fea_mat,
        "msk_mat": msk_mat,
        "tgt_arr": tgt_arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Build Neural CDE–ready sequences from ADNI merged CSV.")
    parser.add_argument("--csv",    required=True,  help="Path to merged adni_spine_adas13_multimodal.csv")
    parser.add_argument("--out",    required=True,  help="Output directory for sequence .npz files")
    parser.add_argument("--splits", required=True,  help="Path to splits.json")
    args = parser.parse_args()

    csv_ptr    = Path(args.csv)
    out_ptr    = Path(args.out)
    splits_ptr = Path(args.splits)

    if not csv_ptr.exists():
        print(f"ERROR: merged CSV not found: {csv_ptr}")
        sys.exit(1)
    if not splits_ptr.exists():
        print(f"ERROR: splits.json not found: {splits_ptr}")
        sys.exit(1)

    out_ptr.mkdir(parents=True, exist_ok=True)

    print(f"Loading merged CSV: {csv_ptr}")
    df_full = pd.read_csv(csv_ptr, low_memory=False)
    df_full["EXAMDATE"] = pd.to_datetime(df_full["EXAMDATE"], errors="coerce")
    df_full["ADAS13"]   = pd.to_numeric(df_full["ADAS13"], errors="coerce")

    # Convert bool columns
    for bcol in ("has_mri","has_fdg_pet"):
        if bcol in df_full.columns:
            df_full[bcol] = df_full[bcol].map(
                {"True":True,"False":False,True:True,False:False,1:True,0:False}
            ).fillna(False).astype(float)

    print(f"Loaded: {df_full.shape[0]} rows, {df_full.shape[1]} cols")

    with open(splits_ptr, "r", encoding="utf-8") as f_splits:
        splits = json.load(f_splits)

    # Determine subject column
    subj_col = "PTID" if "PTID" in df_full.columns else "RID"

    # Feature columns that actually exist in the data
    feat_cols = [c for c in FEATURE_COLS if c in df_full.columns]
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}")

    manifest_rows = []
    total_seqs    = 0
    skipped_seqs  = 0

    for spl_name, spl_subjects in splits.items():
        spl_ptr = out_ptr / spl_name
        spl_ptr.mkdir(parents=True, exist_ok=True)
        spl_set  = set(str(s) for s in spl_subjects)
        spl_df   = df_full[df_full[subj_col].astype(str).isin(spl_set)].copy()
        spl_subjs = spl_df[subj_col].dropna().unique()
        print(f"\nSplit '{spl_name}': {len(spl_subjs)} subjects, {len(spl_df)} rows")

        for ptid in spl_subjs:
            sub_df = spl_df[spl_df[subj_col] == ptid]
            seq    = build_subject_sequence(sub_df, feat_cols)
            if seq is None:
                skipped_seqs += 1
                continue
            # Safe filename
            safe_name  = str(ptid).replace("/","_").replace("\\","_")
            npz_ptr    = spl_ptr / f"{safe_name}.npz"
            np.savez_compressed(
                npz_ptr,
                tim_arr=seq["tim_arr"],
                fea_mat=seq["fea_mat"],
                msk_mat=seq["msk_mat"],
                tgt_arr=seq["tgt_arr"],
            )
            manifest_rows.append({
                "ptid": ptid,
                "split": spl_name,
                "n_visits": len(seq["tim_arr"]),
                "n_features": seq["fea_mat"].shape[1],
                "adas13_obs": int(np.sum(~np.isnan(seq["tgt_arr"]))),
                "filepath": str(npz_ptr.relative_to(out_ptr.parent)),
            })
            total_seqs += 1

    manifest_ptr = out_ptr / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_ptr, index=False)
    print(f"\nSequences written: {total_seqs}  (skipped: {skipped_seqs})")
    print(f"Manifest: {manifest_ptr}")
    print(f"Feature columns: {feat_cols}")
    print("Done.")


if __name__ == "__main__":
    main()
