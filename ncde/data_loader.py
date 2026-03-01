"""
ncde/data_loader.py
===================
Data loading, normalization, and batching for ADNI Neural CDE sequences.

Each subject has a .npz file with:
  - tim_arr (T,)   : months from baseline
  - fea_mat (T, F) : clinical features
  - msk_mat (T, F) : observation masks (True = observed)
  - tgt_arr (T,)   : ADAS13 target

This module handles:
  1. Loading all .npz files for a given split
  2. Computing train-set normalization statistics
  3. Padding variable-length sequences into fixed-size batches
  4. Building natural cubic spline coefficients for the CDE control path
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1. Raw sequence loading
# ---------------------------------------------------------------------------

def load_split_sequences(seq_dir: str, split: str) -> list[dict]:
    """
    Load all .npz files for a given split (train/val/test).
    Returns a list of dicts, each with keys: time, features, mask, target, ptid.
    """
    split_dir = Path(seq_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    sequences = []
    for npz_path in sorted(split_dir.glob("*.npz")):
        data = np.load(npz_path)
        seq_len = len(data["tim_arr"])

        # Skip subjects with fewer than 2 time points (can't form a trajectory)
        if seq_len < 2:
            continue

        sequences.append({
            "time": data["tim_arr"].astype(np.float32),        # (T,)
            "features": data["fea_mat"].astype(np.float32),    # (T, F)
            "mask": data["msk_mat"].astype(np.float32),        # (T, F) as float for JAX
            "target": data["tgt_arr"].astype(np.float32),      # (T,)
            "ptid": npz_path.stem,
        })
    return sequences


# ---------------------------------------------------------------------------
# 2. Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Z-score normalizer fitted on training data.
    Stores per-feature mean and std for features and per-target mean/std.
    """

    def __init__(self):
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.tgt_mean: float = 0.0
        self.tgt_std: float = 1.0
        self.time_scale: float = 1.0  # scale time to [0, 1] range

    def fit(self, sequences: list[dict]) -> "Normalizer":
        """Compute mean/std from a list of training sequences."""
        # Collect all observed values per feature
        all_features = []
        all_targets = []
        all_times = []

        for seq in sequences:
            mask = seq["mask"]  # (T, F)
            feat = seq["features"]  # (T, F)
            tgt = seq["target"]  # (T,)
            tim = seq["time"]  # (T,)

            all_features.append(feat)
            all_targets.append(tgt)
            all_times.append(tim)

        # Stack and compute statistics using masked values
        cat_feat = np.concatenate(all_features, axis=0)  # (N, F)
        cat_mask = np.concatenate([s["mask"] for s in sequences], axis=0)  # (N, F)
        cat_tgt = np.concatenate(all_targets, axis=0)  # (N,)
        cat_time = np.concatenate(all_times, axis=0)  # (N,)

        # Per-feature mean/std (only over observed values)
        self.feat_mean = np.zeros(cat_feat.shape[1], dtype=np.float32)
        self.feat_std = np.ones(cat_feat.shape[1], dtype=np.float32)

        for col_idx in range(cat_feat.shape[1]):
            observed_mask = cat_mask[:, col_idx] > 0.5
            if observed_mask.sum() > 1:
                vals = cat_feat[observed_mask, col_idx]
                self.feat_mean[col_idx] = vals.mean()
                self.feat_std[col_idx] = max(vals.std(), 1e-6)

        # Target statistics (ADAS13)
        valid_tgt = cat_tgt[~np.isnan(cat_tgt)]
        self.tgt_mean = float(valid_tgt.mean()) if len(valid_tgt) > 0 else 0.0
        self.tgt_std = max(float(valid_tgt.std()), 1e-6)

        # Time scale: max time across all subjects
        self.time_scale = max(float(np.nanmax(cat_time)), 1.0)

        return self

    def transform_seq(self, seq: dict) -> dict:
        """Normalize a single sequence dict in-place-style (returns new dict)."""
        feat = (seq["features"] - self.feat_mean) / self.feat_std
        feat = np.nan_to_num(feat, nan=0.0)
        # Zero out unobserved features (mask == 0)
        feat = feat * seq["mask"]

        tgt = (seq["target"] - self.tgt_mean) / self.tgt_std
        tim = seq["time"] / self.time_scale
        tim = np.nan_to_num(tim, nan=0.0)

        return {
            "time": tim.astype(np.float32),
            "features": feat.astype(np.float32),
            "mask": seq["mask"].astype(np.float32),
            "target": tgt.astype(np.float32),
            "ptid": seq["ptid"],
        }


# ---------------------------------------------------------------------------
# 3. Padding and batching
# ---------------------------------------------------------------------------

def pad_and_collate(
    sequences: list[dict],
    max_len: Optional[int] = None,
) -> dict:
    """
    Pad a list of sequences to the same length and stack into arrays.

    Returns dict with:
      - time:     (B, T)
      - features: (B, T, F)
      - mask:     (B, T, F)
      - target:   (B, T)
      - lengths:  (B,) — actual sequence lengths before padding
    """
    if max_len is None:
        max_len = max(len(s["time"]) for s in sequences)

    n_feat = sequences[0]["features"].shape[1]
    batch_size = len(sequences)

    time_pad = np.zeros((batch_size, max_len), dtype=np.float32)
    feat_pad = np.zeros((batch_size, max_len, n_feat), dtype=np.float32)
    mask_pad = np.zeros((batch_size, max_len, n_feat), dtype=np.float32)
    tgt_pad = np.zeros((batch_size, max_len), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)

    for idx, seq in enumerate(sequences):
        seq_len = len(seq["time"])
        actual_len = min(seq_len, max_len)
        time_pad[idx, :actual_len] = seq["time"][:actual_len]
        feat_pad[idx, :actual_len] = seq["features"][:actual_len]
        mask_pad[idx, :actual_len] = seq["mask"][:actual_len]
        tgt_pad[idx, :actual_len] = seq["target"][:actual_len]
        lengths[idx] = actual_len

    return {
        "time": jnp.array(time_pad),
        "features": jnp.array(feat_pad),
        "mask": jnp.array(mask_pad),
        "target": jnp.array(tgt_pad),
        "lengths": jnp.array(lengths),
    }


def make_batches(
    sequences: list[dict],
    batch_size: int,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_len: Optional[int] = None,
) -> list[dict]:
    """
    Split sequences into padded batches.

    Args:
        sequences: list of normalized sequence dicts
        batch_size: number of sequences per batch
        shuffle: whether to shuffle before batching
        rng: numpy random generator for shuffling
        max_len: optional maximum sequence length for padding

    Returns:
        list of batch dicts (each from pad_and_collate)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = np.arange(len(sequences))
    if shuffle:
        rng.shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_seqs = [sequences[i] for i in batch_idx]
        batches.append(pad_and_collate(batch_seqs, max_len=max_len))

    return batches


# ---------------------------------------------------------------------------
# 4. High-level data pipeline
# ---------------------------------------------------------------------------

def build_datasets(
    seq_dir: str,
    batch_size: int = 32,
    max_len: Optional[int] = None,
    subset: Optional[int] = None,
) -> tuple[list[dict], list[dict], list[dict], Normalizer]:
    """
    End-to-end data pipeline: load → normalize → batch.

    Args:
        seq_dir: path to outputs/sequences
        batch_size: batch size
        max_len: max sequence length (None = auto from data)
        subset: if set, only use this many subjects per split (for debugging)

    Returns:
        (train_batches, val_batches, test_batches, normalizer)
    """
    print(f"Loading data from {seq_dir} ...")
    train_seqs = load_split_sequences(seq_dir, "train")
    val_seqs = load_split_sequences(seq_dir, "val")
    test_seqs = load_split_sequences(seq_dir, "test")

    if subset is not None:
        train_seqs = train_seqs[:subset]
        val_seqs = val_seqs[: max(subset // 4, 8)]
        test_seqs = test_seqs[: max(subset // 4, 8)]

    print(f"  Train: {len(train_seqs)} subjects")
    print(f"  Val:   {len(val_seqs)} subjects")
    print(f"  Test:  {len(test_seqs)} subjects")

    # Compute normalization stats from training data
    normalizer = Normalizer().fit(train_seqs)
    print(f"  Feature means: {normalizer.feat_mean}")
    print(f"  Target mean={normalizer.tgt_mean:.2f}, std={normalizer.tgt_std:.2f}")
    print(f"  Time scale: {normalizer.time_scale:.1f} months")

    # Normalize all splits
    train_seqs = [normalizer.transform_seq(s) for s in train_seqs]
    val_seqs = [normalizer.transform_seq(s) for s in val_seqs]
    test_seqs = [normalizer.transform_seq(s) for s in test_seqs]

    # Determine global max length if not specified
    if max_len is None:
        all_lens = [len(s["time"]) for s in train_seqs + val_seqs + test_seqs]
        max_len = max(all_lens)
        print(f"  Max sequence length: {max_len}")

    # Build batches
    rng = np.random.default_rng(42)
    train_batches = make_batches(train_seqs, batch_size, shuffle=True, rng=rng, max_len=max_len)
    val_batches = make_batches(val_seqs, batch_size, shuffle=False, max_len=max_len)
    test_batches = make_batches(test_seqs, batch_size, shuffle=False, max_len=max_len)

    print(f"  Batches: train={len(train_batches)}, val={len(val_batches)}, test={len(test_batches)}")
    print(f"  Feature dim: {train_seqs[0]['features'].shape[1]}")

    return train_batches, val_batches, test_batches, normalizer
