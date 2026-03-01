"""
evaluate.py
===========
Evaluate saved NCDE checkpoints on test and validation sets.
Prints a comparison table of metrics.

Usage:
    python evaluate.py
"""

import os
os.environ["WANDB_MODE"] = "disabled"

import sys
sys.path.insert(0, os.path.dirname(__file__))

import jax
import equinox as eqx

from ncde.data_loader import build_datasets
from ncde.model import create_model
from wandb_train import evaluate_model


def main():
    seq_dir = os.path.join(os.path.dirname(__file__), "outputs", "sequences")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ncde", "checkpoints")

    print("Loading data...")
    train_b, val_b, test_b, normalizer = build_datasets(seq_dir, batch_size=32)
    feature_dim = int(train_b[0]["features"].shape[2])
    key = jax.random.PRNGKey(42)

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline_path = os.path.join(ckpt_dir, "best_baseline.eqx")
    if os.path.exists(baseline_path):
        print("\nEvaluating BASELINE...")
        model = create_model("baseline", feature_dim=feature_dim, hidden_dim=128, vf_width=256, key=key)
        model = eqx.tree_deserialise_leaves(baseline_path, model)
        results["baseline_val"] = evaluate_model(model, val_b, normalizer)
        results["baseline_test"] = evaluate_model(model, test_b, normalizer)
    else:
        print(f"No baseline checkpoint found at {baseline_path}")

    # ── Multimodal ────────────────────────────────────────────────────────
    multi_path = os.path.join(ckpt_dir, "best_multimodal.eqx")
    if os.path.exists(multi_path):
        print("Evaluating MULTIMODAL...")
        model = create_model("multimodal", feature_dim=feature_dim, hidden_dim=128, embed_dim=8, vf_width=256, key=key)
        model = eqx.tree_deserialise_leaves(multi_path, model)
        results["multi_val"] = evaluate_model(model, val_b, normalizer)
        results["multi_test"] = evaluate_model(model, test_b, normalizer)
    else:
        print(f"No multimodal checkpoint found at {multi_path}")

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ADNI ADAS13 Prediction — Model Comparison")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Baseline Val':>14} {'Baseline Test':>14} {'Multi Val':>14} {'Multi Test':>14}")
    print(f"  {'-'*64}")

    for metric in ["mae", "rmse", "r2", "corr"]:
        vals = []
        for key_name in ["baseline_val", "baseline_test", "multi_val", "multi_test"]:
            if key_name in results:
                vals.append(f"{results[key_name][metric]:>14.4f}")
            else:
                vals.append(f"{'N/A':>14}")
        print(f"  {metric.upper():<12} {'  '.join(vals)}")

    for key_name in ["baseline_test", "multi_test"]:
        if key_name in results:
            print(f"\n  {key_name}: {results[key_name]['n_predictions']} predictions")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
